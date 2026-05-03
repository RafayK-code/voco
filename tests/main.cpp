#include <doctest/doctest.h>
#include "test_context.h"

void cleanupTestPipelines();

int main(int argc, char** argv)
{
    TestContext ctx;
    g_ctx = &ctx;

    doctest::Context context(argc, argv);
    int result = context.run();

    cleanupTestPipelines();
    g_ctx = nullptr;
    return result;
}
