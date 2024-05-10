#include <cstdio>

#include "common.h"
#include "multiply.h"
#include "outdeg.h"
#include "encode.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(std::string, input_path, "", "Input file path");
ABSL_FLAG(std::string, input_vector_path, "", "Input vector path");
ABSL_FLAG(std::string, output_vector_path, "", "Output vector path");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    // Ensure that encoder-only flags are recognized by the decoder too.
    (void) absl::GetFlag(FLAGS_allow_random_access);
    (void) absl::GetFlag(FLAGS_greedy_random_access);

    //data
    FILE *in = fopen(absl::GetFlag(FLAGS_input_path).c_str(), "r");
    ZKR_ASSERT(in);

    fseek(in, 0, SEEK_END);
    size_t len = ftell(in);
    fseek(in, 0, SEEK_SET);

    std::vector<uint8_t> data(len);
    ZKR_ASSERT(fread(data.data(), 1, len, in) == len);

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

    //normalise input vector
    {
        double xsum = 0;
        for(size_t c=0; c<nnodes; ++c) {
            xsum += outvec[c];
        }
        for(size_t c=0; c<nnodes; ++c) {
            outvec[c] /= xsum;
        }
    }

    constexpr size_t NITERS = 8;
    constexpr double ALPHA = 0.3;

    //business logic

    //computing outdegree
    if (!zuckerli::ComputeOutDeg(data, outdeg)) {
        fprintf(stderr, "Invalid graph\n");
        return EXIT_FAILURE;
    }

    //starting the loop
    for(size_t iter=0; iter < NITERS; ++iter) {
        //swap invec & outvec
        {
            std::vector<double> tmp = std::move(outvec);
            outvec = std::move(invec);
            invec = std::move(tmp);
        }

        double contrib_dn = 0;
        for(size_t x=0; x<nnodes; ++x) {
            contrib_dn += (outdeg[x] == 0) * invec[x]; //contribution of dangling nodes
            invec[x] /= outdeg[x]; //divide input vector by outdegree
            outvec[x] = 0x0; //0-init
        }
        contrib_dn /= nnodes;

        //multiplication
        if (!zuckerli::DecodeGraph(data, invec, outvec)) {
            fprintf(stderr, "Invalid graph\n");
            return EXIT_FAILURE;
        }

        for (size_t r = 0; r < nnodes; ++r) {
            outvec[r] += contrib_dn; //dangling nodes
            outvec[r] = (1 - ALPHA) * outvec[r] + ALPHA / nnodes; //teleporting
        }

    }

    //outfile
    FILE *out_outvec = fopen(absl::GetFlag(FLAGS_output_vector_path).c_str(), "wb");  // Open in binary format
    ZKR_ASSERT(out_outvec);
    fwrite(outvec.data(), sizeof(double), outvec.size(), out_outvec);
    fclose(out_outvec);  // Close the file

    return EXIT_SUCCESS;
}
