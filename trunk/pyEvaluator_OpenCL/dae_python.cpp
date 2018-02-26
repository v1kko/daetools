#include "stdafx.h"
#include "../Evaluator_OpenCL/cs_opencl.h"
#if defined(__MACH__) || defined(__APPLE__)
#include <python.h>
#endif
#include <string>
#include <boost/python.hpp>
using namespace boost::python;

template<typename ITEM>
boost::python::list getListFromVectorByValue(std::vector<ITEM>& arrItems)
{
    boost::python::list l;

    for(size_t i = 0; i < arrItems.size(); i++)
        l.append(arrItems[i]);

    return l;
}

static boost::python::list lAvailableOpenCLDevices()
{
    std::vector<openclDevice_t> arrDevices = AvailableOpenCLDevices();
    return getListFromVectorByValue(arrDevices);
}

static boost::python::list lAvailableOpenCLPlatforms()
{
    std::vector<openclPlatform_t> arrPlatforms = AvailableOpenCLPlatforms();
    return getListFromVectorByValue(arrPlatforms);
}

static adComputeStackEvaluator_t* CreateComputeStackEvaluator_m(boost::python::list ldevices,
                                                                std::string buildProgramOptions = "")
{
    std::vector<int>    platforms;
    std::vector<int>    devices;
    std::vector<double> taskPortions;

    boost::python::ssize_t i;
    boost::python::ssize_t n = boost::python::len(ldevices);
    for(i = 0; i < n; i++)
    {
        boost::python::tuple dev = boost::python::extract<boost::python::tuple>(ldevices[i]);
        if(boost::python::len(dev) != 3)
            throw std::runtime_error("Invalid device tuple size");

        int platformID     = boost::python::extract<int>(dev[0]);
        int deviceID       = boost::python::extract<int>(dev[1]);
        double taskPortion = boost::python::extract<double>(dev[2]);

        platforms.push_back(platformID);
        devices.push_back(deviceID);
        taskPortions.push_back(taskPortion);
    }

    return CreateComputeStackEvaluator_multi(platforms, devices, taskPortions, buildProgramOptions);
}


// Temporary workaround for Visual Studio 2015 update 3
//  Error   LNK2019 unresolved external symbol "class ClassName const volatile * __cdecl boost::get_pointer<class ClassName const volatile *>(...)
#if _MSC_VER == 1900
#if (defined(_WIN32) || defined(WIN32) || defined(WIN64) || defined(_WIN64))
namespace boost
{
#define POINTER_CONVERSION(CLASS_NAME)   template <> CLASS_NAME const volatile * get_pointer(class CLASS_NAME const volatile *c) {return c;}

POINTER_CONVERSION(adComputeStackEvaluator_t)
}
#endif
#endif


BOOST_PYTHON_MODULE(pyEvaluator_OpenCL)
{
    docstring_options doc_options(true, true, false);

    class_<computestack::adComputeStackEvaluator_t, boost::noncopyable>("adComputeStackEvaluator_t", no_init)
    ;

    class_<openclPlatform_t>("openclPlatform_t", no_init)
        .def_readonly("PlatformID",	&openclPlatform_t::PlatformID)
        .def_readonly("Name",       &openclPlatform_t::Name)
        .def_readonly("Vendor",     &openclPlatform_t::Vendor)
        .def_readonly("Version",	&openclPlatform_t::Version)
        .def_readonly("Profile",	&openclPlatform_t::Profile)
        .def_readonly("Extensions",	&openclPlatform_t::Extensions)
    ;

    class_<openclDevice_t>("openclDevice_t", no_init)
        .def_readonly("PlatformID",         &openclDevice_t::PlatformID)
        .def_readonly("DeviceID",           &openclDevice_t::DeviceID)
        .def_readonly("Name",               &openclDevice_t::Name)
        .def_readonly("DeviceVersion",      &openclDevice_t::DeviceVersion)
        .def_readonly("DriverVersion",      &openclDevice_t::DriverVersion)
        .def_readonly("OpenCLVersion",      &openclDevice_t::OpenCLVersion)
        .def_readonly("MaxComputeUnits",	&openclDevice_t::MaxComputeUnits)
    ;

    def("AvailableOpenCLDevices",      &lAvailableOpenCLDevices);
    def("AvailableOpenCLPlatforms",    &lAvailableOpenCLPlatforms);
    def("CreateComputeStackEvaluator", &CreateComputeStackEvaluator,
                                       ( arg("platformID"), arg("deviceID"), arg("buildProgramOptions") = ""  ), return_value_policy<manage_new_object>());
    def("CreateComputeStackEvaluator", &CreateComputeStackEvaluator_m,
                                       ( arg("devices"), arg("buildProgramOptions") = ""  ), return_value_policy<manage_new_object>());
}
