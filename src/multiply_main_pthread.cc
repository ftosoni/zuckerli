// --input_path web-edu --input_vector_path invec3032 --output_vector_path web-edu.zkr.out --par_degree 4
// /home/tosoni/CLionProjects/zuckerli/gen_graphs

#include <cstdio>

#include "common.h"
#include "multiply.h"
#include "encode.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <thread>

ABSL_FLAG(std::string, input_path, "", "Input file path");
ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");
ABSL_FLAG(std::string, par_degree, "", "Parallelism degree");

void mutiply_part(const std::vector<uint8_t>& data, std::vector<double>& invec, std::vector<double>& outvec) {
    if (!zuckerli::DecodeGraph(data, invec, outvec)) {
        fprintf(stderr, "Invalid graph\n");
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    // Ensure that encoder-only flags are recognized by the decoder too.
    (void) absl::GetFlag(FLAGS_allow_random_access);
    (void) absl::GetFlag(FLAGS_greedy_random_access);

    //args
    const size_t NT = atoi(absl::GetFlag(FLAGS_par_degree).c_str());
    assert(NT);

    //params
    std::vector<std::thread> threads;

    //data
    std::vector<std::vector<uint8_t>> datavec;
    datavec.reserve(NT);

    for(size_t tid=0; tid<NT; ++tid) {
        std::string infilepath;
        infilepath += absl::GetFlag(FLAGS_input_path);
        infilepath += ".";
        infilepath += std::to_string(NT);
        infilepath += ".";
        infilepath += std::to_string(tid);
        infilepath += ".zkr";

//        std::cout << infilepath << std::endl;

        FILE *in = fopen(infilepath.c_str(), "r");
        ZKR_ASSERT(in);

        fseek(in, 0, SEEK_END);
        size_t len = ftell(in);
        fseek(in, 0, SEEK_SET);

        datavec.emplace_back(len);
        ZKR_ASSERT(fread(datavec[tid].data(), 1, len, in) == len);

        fclose(in);
    }

    //invec
    FILE *in_invec = fopen(absl::GetFlag(FLAGS_input_vector_path).c_str(), "r");
    ZKR_ASSERT(in_invec);

    fseek(in_invec, 0, SEEK_END);
    size_t len_invec = ftell(in_invec);
    fseek(in_invec, 0, SEEK_SET);

    std::vector<double> invec(len_invec);
    ZKR_ASSERT(fread(invec.data(), 1, len_invec, in_invec) == len_invec);

    //outvec
    std::vector<double> outvec;

    //multiplication
    for (size_t tid = 0; tid < NT; ++tid) {
        threads.emplace_back(mutiply_part, datavec[tid], std::ref(invec), std::ref(outvec));
    }
    for (size_t tid = 0; tid < NT; ++tid) {
        threads[tid].join();
    }

    //outfile
    FILE *out_outvec = fopen(absl::GetFlag(FLAGS_output_vector_path).c_str(), "wb");  // Open in binary format
    ZKR_ASSERT(out_outvec);
    fwrite(outvec.data(), sizeof(double), outvec.size(), out_outvec);
    fclose(out_outvec);  // Close the file

    return EXIT_SUCCESS;
}
