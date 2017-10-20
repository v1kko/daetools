#include <iostream>
#include "fmi_component.h"
#include <ctime>
#include <boost/lexical_cast.hpp>

int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: fmi_ws_test 'full path to resource directory'\n");
        return 0;
    }
    std::srand(std::time(0));
    fmi2Component c;
    std::string instanceName = "test-" + boost::lexical_cast<std::string>(std::rand());
    std::string guid = "dee8cf5e-9df1-11e7-8f29-680715e7b846";
    std::string resourceLocation = argv[1];
    c = fmi2Instantiate(instanceName.c_str(), fmi2CoSimulation, guid.c_str(), resourceLocation.c_str(), NULL, fmi2False, fmi2False);

    daeFMIComponent_t* comp = (daeFMIComponent_t*)c;
    if(comp == NULL)
        return fmi2Fatal;

    std::vector<fmi2ValueReference> references;
    std::vector<std::string> names;
    std::vector<double> values;
    int N = 0;
    for(std::map<int, fmiObject>::iterator iter = comp->m_FMI_Interface.begin(); iter != comp->m_FMI_Interface.end(); iter++)
    {
        fmiObject& obj = iter->second;
        if(obj.type == "Output" || obj.type == "Local")
        {
            references.push_back(obj.reference);
            names.push_back(obj.name);
            values.push_back(0.0);
            N++;
        }
    }

    fmi2Real t_current = comp->startTime;
    fmi2Real t_step    = comp->step;
    fmi2Real t_horizon = comp->stopTime;
    if(fmi2SetupExperiment(c, false, 1e-5, t_current, false, t_horizon) != fmi2OK)
        exit(-1);

    if(fmi2EnterInitializationMode(c) != fmi2OK)
        exit(-1);

    if(fmi2ExitInitializationMode(c) != fmi2OK)
        exit(-1);

    printf("time");
    for(int i = 0; i < N; i++)
        printf(", %s", names[i].c_str());
    printf("\n");

    while(t_current < t_horizon)
    {
        if(fmi2DoStep(c, t_current, t_step, false) != fmi2OK)
            exit(-1);

        t_current += t_step;

        if(fmi2GetReal(c, references.data(), N, values.data()) != fmi2OK)
            exit(-1);

        printf("%.14f", t_current);
        for(int i = 0; i < N; i++)
            printf(", %.14f", values[i]);
        printf("\n");
    }

    if(fmi2Terminate(c) != fmi2OK)
        exit(-1);
    fmi2FreeInstance(c);

    return 0;
}
