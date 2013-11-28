#ifndef DEAL_II_DATA_REPORTER_H
#define DEAL_II_DATA_REPORTER_H

#include "../Core/coreimpl.h"






#include "../Core/datareporting.h"
#include <boost/filesystem.hpp>
#include <boost/function.hpp>
#include <boost/algorithm/string.hpp>

namespace dae
{
namespace fe_solver
{
using namespace dealii;
using namespace dae::datareporting;

typedef boost::function<void (const std::string&, const std::string&, double*, unsigned int)>  fnProcessSolution;

/*********************************************************************
   dealIIDataReporter
*********************************************************************/
class dealIIDataReporter : public daeDataReporter_t
{
public:
    dealIIDataReporter(const fnProcessSolution& callback, const std::map<std::string, size_t>& mapVariables)
        : m_callback(callback), m_outputCounter(0), m_mapVariables(mapVariables)
    {
    }

    virtual ~dealIIDataReporter(void)
    {
    }

public:
    std::string GetName() const
    {
        return "DealIIDataReporter";
    }

    bool Connect(const string& strConnectString, const string& strProcessName)
    {
        m_strOutputDirectory = strConnectString;

        // Check if output directory exists
        // If not create the whole hierarchy, up to the top directory
        boost::filesystem::path folder(m_strOutputDirectory);
        if(!boost::filesystem::exists(folder))
        {
            if(!boost::filesystem::create_directories(folder))
            {
                daeDeclareException(exInvalidCall);
                e << "Invalid output directory name (" << m_strOutputDirectory
                  << ") specified as a 'connection string representing an output directory for daeDealIIDataReporter";
                throw e;
            }
        }

        return true;
    }

    bool Disconnect(void)
    {
        return true;
    }

    bool IsConnected(void)
    {
        return true;
    }

    bool StartRegistration(void)
    {
        return true;
    }

    bool RegisterDomain(const daeDataReporterDomain* pDomain)
    {
        return true;
    }

    bool RegisterVariable(const daeDataReporterVariable* pVariable)
    {
        std::vector<std::string> names;
        boost::algorithm::split(names, pVariable->m_strName, boost::algorithm::is_any_of("."));
        std::string strVariableName = names[names.size()-1];

        boost::filesystem::path vtkPath(m_strOutputDirectory);
        boost::filesystem::path variableFolder(strVariableName);
        vtkPath /= variableFolder;

        if(!boost::filesystem::exists(vtkPath.c_str()))
        {
            if(!boost::filesystem::create_directories(vtkPath.c_str()))
            {
                daeDeclareException(exInvalidCall);
                e << "Cannot create an output directory for the variable: " << strVariableName;
                throw e;
            }
        }

        return true;
    }

    bool EndRegistration(void)
    {
        return true;
    }

    bool StartNewResultSet(real_t dTime)
    {
        m_dCurrentTime = dTime;
        m_outputCounter++;
        return true;
    }

    bool EndOfData(void)
    {
        return true;
    }

    bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
    {
        std::vector<std::string> names;
        boost::algorithm::split(names, pVariableValue->m_strName, boost::algorithm::is_any_of("."));
        std::string strVariableName = names[names.size()-1];

        // Do not report variables that do not belong to this finite element object
        if(m_mapVariables.find(strVariableName) == m_mapVariables.end())
            return true;

        // Build the new file name
        boost::filesystem::path vtkFilename((boost::format("%05d.%s(t=%f).vtk") % m_outputCounter % strVariableName % m_dCurrentTime).str());
        boost::filesystem::path vtkPath(m_strOutputDirectory);
        boost::filesystem::path variableFolder(strVariableName);
        vtkPath /= variableFolder;
        vtkPath /= vtkFilename;

        // Call function in the finite element object to perform some processing and to save values in a file
        m_callback(vtkPath.c_str(), strVariableName, pVariableValue->m_pValues, pVariableValue->m_nNumberOfPoints);

        return true;
    }

public:
    real_t                          m_dCurrentTime;
    int                             m_outputCounter;
    fnProcessSolution               m_callback;
    std::string                     m_strOutputDirectory;
    std::map<std::string, size_t>   m_mapVariables;
};


}
}

#endif
