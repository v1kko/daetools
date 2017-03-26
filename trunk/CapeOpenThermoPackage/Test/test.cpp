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
    std::vector<std::string> strarrCompoundIDs(4), strarrCompoundCASNumbers;
    strarrCompoundIDs[0] = "Hydrogen";
    strarrCompoundIDs[1] = "Carbon monoxide";
    strarrCompoundIDs[2] = "Methane";
    strarrCompoundIDs[3] = "Carbon dioxide";

    std::map<std::string, daeeThermoPackagePhase> mapAvailablePhases;
    mapAvailablePhases["Vapor"] = eVapor;

    dae::tpp::daeThermoPhysicalPropertyPackage_t* package = daeCreateCapeOpenPropertyPackage();
    package->LoadPackage("ChemSep Property Package Manager", "SMROG", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);
    //package->LoadPackage("TEA Property Package Manager", "SMROG", strarrCompoundIDs, strarrCompoundCASNumbers, mapAvailablePhases);

    double result;
    std::vector<double> results;
    std::vector<double> fraction(4);
    double pressure = 1e5;
    double temperature = 300;
    fraction[0] = 0.7557;
    fraction[1] = 0.0400;
    fraction[2] = 0.0350;
    fraction[3] = 0.1693;

    std::wcout << "*****************************************************************" << std::endl;
    std::wcout << "                         Single phase tests                      " << std::endl;
    std::wcout << "*****************************************************************" << std::endl;

    try
    {
        std::wcout << "TEST 1. Hydrogen heatOfFusionAtNormalFreezingPoint is: ";
        result = package->PureCompoundConstantProperty("heatOfFusionAtNormalFreezingPoint", "Hydrogen");
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
        result = package->PureCompoundTDProperty("idealGasEnthalpy", temperature, "Methane");
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
        result = package->PureCompoundPDProperty("boilingPointTemperature", pressure, "Hydrogen");
        std::wcout << result << " K" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 4. SMROG mixture density is: ";
        result = package->SinglePhaseScalarProperty("density", pressure, temperature, fraction, "Vapor", eMole);
        std::wcout << result << " mol/m3" << std::endl << std::endl;
    }
    catch (std::exception& e)
    {
        std::wcout << "CALCULATION FAILED" << std::endl << std::endl;
        std::cout << e.what() << std::endl;
    }

    try
    {
        std::wcout << "TEST 5. SMROG mixture fugacity is: ";
        package->SinglePhaseVectorProperty("fugacity", pressure, temperature, fraction, "Vapor", results, eUndefinedBasis);
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

    daeDeleteCapeOpenPropertyPackage(package);
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
        result = package->TwoPhaseScalarProperty("surfaceTension",
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
        package->TwoPhaseVectorProperty("kvalue",
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

int main()
{
    ::CoInitialize(NULL);

    test_single_phase();
    test_two_phase();

    ::CoUninitialize();
}
