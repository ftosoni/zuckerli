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

inline static void set_core(std::thread *tid, int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(tid->native_handle(),sizeof(cpu_set_t), &cpuset);
    return;
};

ABSL_FLAG(std::string, input_path, "", "Input file path");
//ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
//ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");
ABSL_FLAG(std::string, ccount_path, "", "Column count");

ABSL_FLAG(std::string, verbose, "0", "verbose");
ABSL_FLAG(std::string, maxiter, "100", "maximum number of iteration, def. 100");
ABSL_FLAG(std::string, dampf, "0.9", "damping factor (default 0.9)");
ABSL_FLAG(std::string, topk, "3", "show top K nodes (default 3)");
ABSL_FLAG(std::string, pardegree, "2", "parallelism degree, def. 2");

static void usage_and_exit(char *name) {
    fprintf(stderr, "Usage:\n\t  %s [options] --input_path matrix_name --ccount_path col_count_file\n", name);
    fprintf(stderr, "\t\t--verbose        verbose, def. 0\n");
    fprintf(stderr,"\t\t--pardegree       parallelism degree, def. 2\n");
    fprintf(stderr, "\t\t--maxiter        maximum number of iteration, def. 100\n");
//    fprintf(stderr,"\t\t-e eps         stop if error<eps (default ignore error)\n");
    fprintf(stderr, "\t\t--dampf          damping factor (default 0.9)\n");
    fprintf(stderr, "\t\t--topk           show top K nodes (default 3)\n");
    exit(1);
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    // Ensure that encoder-only flags are recognized by the decoder too.
    (void) absl::GetFlag(FLAGS_allow_random_access);
    (void) absl::GetFlag(FLAGS_greedy_random_access);

    //args
    const int verbose=atoi(absl::GetFlag(FLAGS_verbose).c_str());
    const int maxiter=atoi(absl::GetFlag(FLAGS_maxiter).c_str());
    const double dampf=atof(absl::GetFlag(FLAGS_dampf).c_str());
    int topk=atoi(absl::GetFlag(FLAGS_topk).c_str());
    const int NT=atoi(absl::GetFlag(FLAGS_pardegree).c_str());
    if(verbose>0) {
        fputs("==== Command line:\n",stderr);
        for(int i=0;i<argc;i++)
            fprintf(stderr," %s",argv[i]);
        fputs("\n",stderr);
    }
    // check command line
    if(maxiter<1 || topk<1) {
        fprintf(stderr,"Error! Options --maxiter and --topk must be at least one\n");
        usage_and_exit(argv[0]);
    }
    if(NT<2) {
        fprintf(stderr,"Error! Option --pardegree must be at least two\n");
        usage_and_exit(argv[0]);
    }
    if(dampf<0 || dampf>1) {
        fprintf(stderr,"Error! Options --dampf must be in the range [0,1]\n");
        usage_and_exit(argv[0]);
    }

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
        set_core(&threads[tid], tid);
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
    for(size_t iter=0; iter < maxiter; ++iter) {
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
            set_core(&threads[tid], tid);       
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
            set_core(&threads[tid], tid);
        }
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid].join();
        }

        //finalisation
        auto finalise_f = [](
                std::vector<double> &outvec,
                const double contrib_dn,
                const size_t nnodes,
                const double dampf,
                const uint8_t NT,
                const uint8_t tid) {
            const size_t row_block_size = (nnodes + NT - 1) / NT;
            const size_t row_bgn = tid * row_block_size;
            const size_t row_end = std::min<size_t>((tid + 1) * row_block_size, nnodes);
            for (size_t r = row_bgn; r < row_end; ++r) {
                outvec[r] += contrib_dn; //dangling nodes
                outvec[r] = dampf * outvec[r] + (1-dampf) / nnodes; //teleporting
            }
        };
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid] = std::thread(finalise_f, std::ref(outvec), contrib_dn, nnodes, dampf, NT, tid);
            set_core(&threads[tid], tid);
        }
        for(uint8_t tid=0; tid<NT; ++tid) {
            threads[tid].join();
        }

    }
    free(contrib_dn_helper);

//    for(auto const &e : outvec) std::cout << e << std::endl;

    if(verbose>0) {
        double sum = 0.0;
        for(int i=0;i<nnodes;i++) sum += outvec[i];
        fprintf(stderr,"Sum of ranks: %f (should be 1)\n",sum);
    }

    // retrieve topk nodes
    topk = std::min<int>(topk, nnodes);
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
    // report topk nodes sorted by decreasing rank
    if (verbose>0) {
        fprintf(stderr, "Top %d ranks:\n",topk);
        for(int i=0;i<topk;i++) fprintf(stderr,"  %d %lf\n",top[i],outvec[top[i]]);
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
