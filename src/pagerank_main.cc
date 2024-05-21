#include <cstdio>
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

ABSL_FLAG(std::string, verbose, "0", "verbose");
ABSL_FLAG(std::string, maxiter, "100", "maximum number of iteration, def. 100");
ABSL_FLAG(std::string, dampf, "0.9", "damping factor (default 0.9)");
ABSL_FLAG(std::string, topk, "3", "show top K nodes (default 3)");
//ABSL_FLAG(std::string, pardegree, "2", "parallelism degree, def. 2");

static void usage_and_exit(char *name)
{
    fprintf(stderr,"Usage:\n\t  %s [options] --input_path matrix_name.zkr --ccount_path col_count_file\n",name);
    fprintf(stderr,"\t\t--verbose        verbose, def. 0\n");
//    fprintf(stderr,"\t\t--pardegree       parallelism degree, def. 2\n");
    fprintf(stderr,"\t\t--maxiter        maximum number of iteration, def. 100\n");
//    fprintf(stderr,"\t\t-e eps         stop if error<eps (default ignore error)\n");
    fprintf(stderr,"\t\t--dampf          damping factor (default 0.9)\n");
    fprintf(stderr,"\t\t--topk           show top K nodes (default 3)\n");
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
//    const int pardegree=atoi(absl::GetFlag(FLAGS_pardegree).c_str());
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
    if(dampf<0 || dampf>1) {
        fprintf(stderr,"Error! Options --dampf must be in the range [0,1]\n");
        usage_and_exit(argv[0]);
    }

    //data
    FILE *in = fopen(absl::GetFlag(FLAGS_input_path).c_str(), "r");
    ZKR_ASSERT(in);

    fseek(in, 0, SEEK_END);
    size_t len = ftell(in);
    fseek(in, 0, SEEK_SET);

    std::vector<uint8_t> data(len);
    ZKR_ASSERT(fread(data.data(), 1, len, in) == len);

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

    //computing outdegree
//    if (!zuckerli::ComputeOutDeg(data, outdeg)) {
//        fprintf(stderr, "Invalid graph\n");
//        return EXIT_FAILURE;
//    }

    //starting the loop
    for(size_t iter=0; iter < maxiter; ++iter) {
        //swap invec & outvec
        {
            std::vector<double> tmp = std::move(outvec);
            outvec = std::move(invec);
            invec = std::move(tmp);
        }

        double contrib_dn = 0;
        for(size_t x=0; x<nnodes; ++x) {
            contrib_dn += (outdeg[x] == 0) * invec[x]; //contribution of dangling nodes
            if (outdeg[x]) invec[x] /= outdeg[x]; //divide input vector by outdegree
            outvec[x] = 0.0; //0-init
        }
        contrib_dn /= nnodes;

        //multiplication
        if (!zuckerli::DecodeGraph(data, invec, outvec)) {
            fprintf(stderr, "Invalid graph\n");
            return EXIT_FAILURE;
        }

        for (size_t r = 0; r < nnodes; ++r) {
            outvec[r] += contrib_dn; //dangling nodes
            outvec[r] = dampf * outvec[r] + (1-dampf) / nnodes; //teleporting
        }

    }

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
