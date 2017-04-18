// CapeOpenTPP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <comcat.h>
#include <atlcom.h>
#include <iostream>
#include <vector>
#include <map>
#include "../cape_open_package.h"

using dae::tpp::daeeThermoPackagePropertyType;
using dae::tpp::daeeThermoPackagePhase;
using dae::tpp::daeeThermoPackageBasis;
using dae::tpp::daeThermoPhysicalPropertyPackage_t;
using dae::tpp::eMole;
using dae::tpp::eMass;
using dae::tpp::eUndefinedBasis;
using dae::tpp::eVapor;
using dae::tpp::eLiquid;
using dae::tpp::eSolid;

void test_single_phase()
{
    std::vector<std::string> strarrCompoundIDs(3), strarrCompoundCASNumbers;
    strarrCompoundIDs[0] = "Hydrogen";
    strarrCompoundIDs[1] = "Methane";
    strarrCompoundIDs[2] = "Carbon dioxide";

    std::map<std::string, daeeThermoPackagePhase> mapAvailablePhases;
    mapAvailablePhases["Vapor"] = eVapor;

    dae::tpp::daeThermoPhysicalPropertyPackage_t* package = daeCreateCapeOpenPropertyPackage();
    package->LoadPackage("ChemSep Property Package Manager", "H2+CH4+CO2", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);
    //package->LoadPackage("TEA Property Package Manager", "H2+CH4+CO2", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);

    double result;
    std::vector<double> results;
    std::vector<double> fraction(3);
    double pressure = 1e5;
    double temperature = 300;
    fraction[0] = 0.7557;
    fraction[1] = 0.0750;
    fraction[2] = 0.1693;

    std::wcout << "*****************************************************************" << std::endl;
    std::wcout << "                         Single phase tests                      " << std::endl;
    std::wcout << "*****************************************************************" << std::endl;

    try
    {
        std::wcout << "TEST 1. Hydrogen heatOfFusionAtNormalFreezingPoint is: ";
        result = package->GetCompoundConstant("heatOfFusionAtNormalFreezingPoint", "Hydrogen");
        std::wcout << result << " J/mol" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 2. Methane idealGasEnthalpy is: ";
        result = package->GetTDependentProperty("idealGasEnthalpy", temperature, "Methane");
        std::wcout << result << " J/mol" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 3. Hydrogen boilingPointTemperature is: ";
        result = package->GetPDependentProperty("boilingPointTemperature", pressure, "Hydrogen");
        std::wcout << result << " K" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 4. H2+CH4+CO2 mixture density is: ";
        result = package->CalcSinglePhaseScalarProperty("density", pressure, temperature, fraction, "Vapor", eMole);
        std::wcout << result << " mol/m3" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 5. H2+CH4+CO2 mixture fugacity is: ";
        package->CalcSinglePhaseVectorProperty("fugacity", pressure, temperature, fraction, "Vapor", results, eUndefinedBasis);
        std::wcout << "[";
        for (size_t i = 0; i < results.size(); i++)
            std::wcout << (i == 0 ? "" : ", ") << results[i];
        std::wcout << "]" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }
    std::wcout << std::endl << std::endl;

    delete package;
}

void test_two_phase()
{
    std::vector<std::string> strarrCompoundIDs(2), strarrCompoundCASNumbers;
    strarrCompoundIDs[0] = "Water";
    strarrCompoundIDs[1] = "Ethanol";

    std::map<std::string, daeeThermoPackagePhase> mapAvailablePhases;
    mapAvailablePhases["Vapor"]  = eVapor;
    mapAvailablePhases["Liquid"] = eLiquid;

    dae::tpp::daeThermoPhysicalPropertyPackage_t* package = daeCreateCapeOpenPropertyPackage();
    package->LoadPackage("ChemSep Property Package Manager", "Water+Ethanol", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);
    //package->LoadPackage("TEA Property Package Manager", "Water+Ethanol", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);

    double result;
    std::vector<double> results;

    std::vector<double> x1(2);
    double P1 = 1e5;
    double T1 = 300;
    x1[0] = 0.60;
    x1[1] = 0.40;

    std::vector<double> x2(2);
    double P2 = 1e5;
    double T2 = 300;
    x2[0] = 0.60;
    x2[1] = 0.40;

    std::wcout << "*****************************************************************" << std::endl;
    std::wcout << "                         Two phase tests                         " << std::endl;
    std::wcout << "*****************************************************************" << std::endl;
    
    try
    {
        std::wcout << "TEST 1. Water+Ethanol mixture surfaceTension is: ";
        result = package->CalcTwoPhaseScalarProperty("surfaceTension",
                                                     P1, T1, x1, "Liquid", 
                                                     P2, T2, x2, "Vapor",
                                                     eUndefinedBasis);
        std::wcout << result << " N/m" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 2. Water+Ethanol mixture kvalue is: ";
        package->CalcTwoPhaseVectorProperty("kvalue",
                                            P1, T1, x1, "Liquid",
                                            P2, T2, x2, "Vapor",
                                            results,
                                            eUndefinedBasis);
        std::wcout << "[";
        for (size_t i = 0; i < results.size(); i++)
            std::wcout << (i == 0 ? "" : ", ") << results[i];
        std::wcout << "]" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }
}

#include "../../coolprop/include/CoolProp.h"
#include <iostream>

void test_cool_prop()
{
    using namespace CoolProp;

    std::vector< std::vector<double> > matResults;
    std::vector<double> T(1, 298), P(1, 1e5);

    std::vector<std::string> properties;
    properties.push_back("Dmolar");

    std::vector<std::string> compounds;
    compounds.push_back("Hydrogen");
    compounds.push_back("Methane");
    compounds.push_back("CarbonDioxide"); 

    std::vector<double> x;
    x.push_back(0.7550);
    x.push_back(0.0750);
    x.push_back(0.1700);

    std::string strBackend = "HEOS";
    set_debug_level(0);

    //std::cout << "T = " << T[0] << std::endl;
    //std::cout << "P = " << P[0] << std::endl;
    //std::cout << "x = [";
    //for (size_t i = 0; i < x.size(); i++)
    //    std::cout << (i == 0 ? "" : ", ") << x[i];
    //std::cout << "]" << std::endl;
    //std::cout << "compounds = [";
    //for (size_t i = 0; i < compounds.size(); i++)
    //    std::cout << (i == 0 ? "" : ", ") << compounds[i];
    //std::cout << "]" << std::endl;
    //std::cout.flush();

    try
    {
        //std::cout << "Dmolar = " << PropsSI("Dmolar", "T", 298, "P", 1e5, "Hydrogen[0.755]&CarbonMonoxide[0.040]&Methane[0.035]&CarbonDioxide[0.17]") << std::endl;
        //{
        //    std::vector<double> z(2, 0.5);
        //    // Second type (C++ only, a bit faster, allows for vector inputs and outputs)
        //    std::vector<std::string> fluids; fluids.push_back("Propane"); fluids.push_back("Ethane");
        //    std::vector<std::string> outputs; outputs.push_back("Dmolar");
        //    std::vector<double> T(1, 298), p(1, 1e5);
        //    std::cout << PropsSImulti(outputs, "T", T, "P", p, "", fluids, z)[0][0] << std::endl; // Default backend is HEOS
        //    std::cout << PropsSImulti(outputs, "T", T, "P", p, "HEOS", fluids, z)[0][0] << std::endl;
        //}

        // Results in the "One stationary point (not good) for T=... P=..." exception thrown
        matResults = PropsSImulti(properties, "T", T, "P", P, strBackend, compounds, x);
        if (matResults.size() == 1 && matResults[0].size() == 1)
            std::cout << "Dmolar = " << matResults[0][0] << std::endl;
    }
    catch (const std::exception& e) 
    {
        std::cout << e.what() << std::endl;
    }
}

int main()
{
    ::CoInitialize(NULL);

    test_single_phase();
    test_two_phase();
    test_cool_prop();

    ::CoUninitialize();
}
