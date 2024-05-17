#include <cstdio>
#include <thread>
#include <fstream>

#include "common.h"
#include "multiply.h"
#include "outdeg.h"
#include "encode.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "pagerank_utils.h"

ABSL_FLAG(std::string, input_path, "", "Input file path");
//ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
//ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");
ABSL_FLAG(std::string, ccount_path, "", "Column count");
ABSL_FLAG(std::string, par_degree, "", "Parallelism degree");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    // Ensure that encoder-only flags are recognized by the decoder too.
    (void) absl::GetFlag(FLAGS_allow_random_access);
    (void) absl::GetFlag(FLAGS_greedy_random_access);

    //args
    const uint8_t NT = atoi(absl::GetFlag(FLAGS_par_degree).c_str());
    assert(NT);

    //params
    std::vector<std::thread> threads(NT);

    //data
    std::vector<std::vector<uint8_t>> datavec;
    datavec.resize(NT);

    //for(size_t tid=0; tid<NT; ++tid) {
    auto read_data_f = [](
            std::vector<std::vector<uint8_t>> &datavec,
            const uint8_t NT,
            const uint8_t tid) {
        std::string infilepath;
        infilepath += absl::GetFlag(FLAGS_input_path);
        infilepath += ".";
        infilepath += std::to_string(NT);
        infilepath += ".";
        infilepath += std::to_string(tid);
        infilepath += ".zkr";

        FILE *in = fopen(infilepath.c_str(), "r");
        ZKR_ASSERT(in);

        fseek(in, 0, SEEK_END);
        size_t len = ftell(in);
        fseek(in, 0, SEEK_SET);

        datavec[tid].resize(len);
        ZKR_ASSERT(fread(datavec[tid].data(), 1, len, in) == len);

        fclose(in);
    };
    for(uint8_t tid=0; tid<NT; ++tid){
        threads[tid] = std::thread(read_data_f, std::ref(datavec), NT, tid);
    }
    for(uint8_t tid=0; tid<NT; ++tid){
        threads[tid].join();
    }


    //reading column count files
    uint32_t nnodes;
    {
        std::ifstream file(absl::GetFlag(FLAGS_ccount_path).c_str(), std::ios::binary);  // Open the file in binary mode
        if (!file) {
            std::cout << "Failed to open the file." << std::endl;
            return 1;
        }
        // Move the file pointer to the end of the file
        file.seekg(0, std::ios::end);
        // Get the position of the file pointer, which represents the length of the file
        std::streampos length = file.tellg();
        nnodes = length / sizeof(u_int32_t);
        file.close();
    }

    //structures
    std::vector<double> outvec(nnodes, 1.0/nnodes), invec(nnodes);
    std::vector<uint32_t> outdeg(nnodes);
    {
        FILE *outdegfile = fopen(absl::GetFlag(FLAGS_ccount_path).c_str(), "r");
        ZKR_ASSERT(fread(outdeg.data(), sizeof(u_int32_t), nnodes, outdegfile) == nnodes);
        fclose(outdegfile);
    }


    //business logic

    //starting the loop
    double *contrib_dn_helper = (double *) calloc(NT, sizeof(double));
    for(size_t iter=0; iter < NITERS; ++iter) {
        //swap invec & outvec
        {
            std::vector<double> tmp = std::move(outvec);
            outvec = std::move(invec);
            invec = std::move(tmp);
        }

        double contrib_dn = 0;
        auto contrib_dn_f = [](
                double *contrib_dn_helper,
                const uint32_t nnodes,
                std::vector<uint32_t> &outdeg,
                std::vector<double> &invec,
                std::vector<double> &outvec,
                const uint8_t NT,
                const uint8_t tid
        ) {
            contrib_dn_helper[tid] = 0.0;
            const size_t x_block_size = (nnodes + NT - 1) / NT;
            const size_t x_bgn = tid * x_block_size;
            const size_t x_end = std::min<size_t>((tid + 1) * x_block_size, nnodes);
            for (size_t x = x_bgn; x < x_end; x++) {
                contrib_dn_helper[tid] += (outdeg[x] == 0) * invec[x]; //contribution of dangling nodes
                if(outdeg[x]) invec[x] /= outdeg[x]; //divide input vector by outdegree
                outvec[x] = 0.0; //0-init
            }
            contrib_dn_helper[tid] /= nnodes;
        };
        for(uint8_t tid=0; tid<NT; ++tid){
            threads[tid] = std::thread(contrib_dn_f, contrib_dn_helper, nnodes, std::ref(outdeg), std::ref(invec), std::ref(outvec), NT, tid);
        }
        for(uint8_t tid=0; tid<NT; ++tid){
            threads[tid].join();
            contrib_dn += contrib_dn_helper[tid];
        }

        //multiplication
        auto mult_f = [] (
                std::vector<uint8_t> &datavec_component,
                std::vector<double> &invec,
                std::vector<double> &outvec
                ) {
            if (!zuckerli::DecodeGraph(datavec_component, invec, outvec)) {
                fprintf(stderr, "Invalid graph\n");
                //return EXIT_FAILURE;
            }
        };
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid] = std::thread(mult_f, std::ref(datavec[tid]), std::ref(invec), std::ref(outvec));
        }
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid].join();
        }

        //finalisation
        auto finalise_f = [](
                std::vector<double> &outvec,
                const double contrib_dn,
                const size_t nnodes,
                const uint8_t NT,
                const uint8_t tid) {
            const size_t row_block_size = (nnodes + NT - 1) / NT;
            const size_t row_bgn = tid * row_block_size;
            const size_t row_end = std::min<size_t>((tid + 1) * row_block_size, nnodes);
            for (size_t r = row_bgn; r < row_end; ++r) {
                outvec[r] += contrib_dn; //dangling nodes
                outvec[r] = (1 - ALPHA) * outvec[r] + ALPHA / nnodes; //teleporting
            }
        };
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid] = std::thread(finalise_f, std::ref(outvec), contrib_dn, nnodes, NT, tid);
        }
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid].join();
        }

    }
    free(contrib_dn_helper);

//    for(auto const &e : outvec) std::cout << e << std::endl;

    // retrieve topk nodes
    unsigned topk = std::min<unsigned>(TOPK, nnodes);
    unsigned *top = (unsigned *) calloc(topk, sizeof(*top));
    unsigned *aux = (unsigned *) calloc(topk, sizeof(*top));
    if(top==NULL || aux==NULL){
        perror("Cannot allocate topk/aux array");
        exit(-1);
    }
    kLargest(outvec,aux,nnodes,topk);
    // get sorted nodes in top
    for(long int i=topk-1;i>=0;i--) {
        top[i] = aux[0];
        aux[0] = aux[i];
        minHeapify(outvec,aux,i,0);
    }
    // report topk nodes id's only on stdout
    fprintf(stdout,"Top:");
    for(int i=0;i<topk;i++) fprintf(stdout," %d",top[i]);
    fprintf(stdout,"\n");
    //deallocate
    free(top);
    free(aux);

    return EXIT_SUCCESS;
}
