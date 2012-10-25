#include "daetools_model.h"

int main(int argc, char *argv[])
{
    initial_values();
    std::cout << "number_of_roots = " << number_of_roots() << std::endl;
    
    return 0;
}