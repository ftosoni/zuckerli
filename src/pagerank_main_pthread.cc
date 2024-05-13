#include <cstdio>
#include <thread>

#include "common.h"
#include "multiply.h"
#include "outdeg.h"
#include "encode.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, input_path, "", "Input file path");
ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");
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

    //outvec
    FILE *in_invec = fopen(absl::GetFlag(FLAGS_input_vector_path).c_str(), "r");
    ZKR_ASSERT(in_invec);

    fseek(in_invec, 0, SEEK_END);
    size_t len_invec = ftell(in_invec);
    fseek(in_invec, 0, SEEK_SET);

    //structures
    assert(len_invec % sizeof(double) == 0);
    const size_t nnodes = len_invec / sizeof(double);
    std::vector<double> outvec(nnodes), invec(nnodes);
    std::vector<uint32_t> outdeg(nnodes);
    ZKR_ASSERT(fread(outvec.data(), 1, len_invec, in_invec) == len_invec);

    //sum input vector
    double* xsum_helper = (double *) calloc(NT, sizeof(double ));
    auto xsum_f = [](
            std::vector<double> &outvec,
            double *xsum_helper,
            const size_t nnodes,
            const uint8_t NT,
            const uint8_t tid
    ){
        xsum_helper[tid] = 0.0;
        const size_t col_block_size = (nnodes + NT - 1) / NT;
        const size_t col_bgn = tid * col_block_size;
        const size_t col_end = std::min<size_t>((tid + 1) * col_block_size, nnodes);
//        assert(col_end > col_bgn);
        for(size_t c=col_bgn; c<col_end; c++) {
            xsum_helper[tid] += outvec[c];
        }
    };
    double xsum = 0.0;
    for(uint8_t tid=0; tid<NT; ++tid) {
        threads[tid] = std::thread(xsum_f, std::ref(outvec), xsum_helper, nnodes, NT, tid);
    }
    for(uint8_t tid=0; tid<NT; ++tid) {
        threads[tid].join();
        xsum += xsum_helper[tid];
    }
    free(xsum_helper);

    //normalise input vector
    auto xnormalise_f = [](
            std::vector<double> &outvec,
            const double xsum,
            const size_t nnodes,
            const uint8_t NT,
            const uint8_t tid
    ){
        const size_t col_block_size = (nnodes + NT - 1) / NT;
        const size_t col_bgn = tid * col_block_size;
        const size_t col_end = std::min<size_t>((tid + 1) * col_block_size, nnodes);
//        assert(col_end > col_bgn);
        for(size_t c=col_bgn; c<col_end; c++) {
            outvec[c] /= xsum;
        }
    };
    for(uint8_t tid=0; tid<NT; ++tid) {
        threads[tid] = std::thread(xnormalise_f, std::ref(outvec), xsum, nnodes, NT, tid);
    }
    for(uint8_t tid=0; tid<NT; ++tid) {
        threads[tid].join();
    }

    constexpr size_t NITERS = 8;
    constexpr double ALPHA = 0.3;

    //business logic

    //computing outdegree
    std::vector<std::vector<uint32_t>> outdeg_helper(NT-1);
    auto outdegree_f = [](
            std::vector<uint8_t> &datavec_component,
            std::vector<uint32_t> &outdeg_helper_component
            ) {
        if (!zuckerli::ComputeOutDeg(datavec_component, outdeg_helper_component)) {
            fprintf(stderr, "Invalid graph\n");
        }
    };
    //map
    {
        const uint8_t tid = 0; //first thread
        threads[tid] = std::thread(outdegree_f, std::ref(datavec[tid]), std::ref(outdeg));
    }
    for(uint8_t tid=1; tid<NT; ++tid) { //other threads
        threads[tid] = std::thread(outdegree_f, std::ref(datavec[tid]), std::ref(outdeg_helper[tid-1]));
    }
    //reduce
    threads[0].join();
    assert(outdeg.size() == nnodes);
    for(uint8_t tid=1; tid<NT; ++tid) { //other threads
        threads[tid].join();
        assert(outdeg_helper[tid-1].size() == nnodes);
        for(size_t c=0; c<nnodes; ++c) {
            outdeg[c] += outdeg_helper[tid-1][c];
        }
        outdeg_helper[tid-1].clear();
    }
    outdeg_helper.clear();

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

    for(auto const &e : outvec) std::cout << e << std::endl;

    //outfile
    FILE *out_outvec = fopen(absl::GetFlag(FLAGS_output_vector_path).c_str(), "wb");  // Open in binary format
    ZKR_ASSERT(out_outvec);
    fwrite(outvec.data(), sizeof(double), outvec.size(), out_outvec);
    fclose(out_outvec);  // Close the file

    return EXIT_SUCCESS;
}
