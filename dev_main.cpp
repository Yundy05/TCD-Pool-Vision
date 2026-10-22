#include <cstdlib>
#include <exception>
#include <iostream>

void MyApplication();

int main()
{
    try
    {
        MyApplication();
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Application error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}