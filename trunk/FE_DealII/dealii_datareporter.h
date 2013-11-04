#ifndef DEAL_II_DATA_REPORTER_H
#define DEAL_II_DATA_REPORTER_H

#include "../Core/datareporting.h"
#include <boost/filesystem.hpp>
#include <boost/function.hpp>

namespace dae
{
namespace fe_solver
{
using namespace dealii;
using namespace dae::datareporting;

typedef boost::function<void (const std::string&, const std::string&, double*, unsigned int)>  fnProcessSolution;

/*********************************************************************
   daeDealIIDataReporter
*********************************************************************/
class daeDealIIDataReporter : public daeDataReporter_t
{
public:
    daeDealIIDataReporter(const fnProcessSolution& callback)
        : m_callback(callback), m_outputCounter(0)
    {
    }

    virtual ~daeDealIIDataReporter(void)
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
        return true;
    }

    bool EndRegistration(void)
    {
        return true;
    }

    bool StartNewResultSet(real_t dTime)
    {
        m_dCurrentTime = dTime;
        return true;
    }

    bool EndOfData(void)
    {
        return true;
    }

    bool SendVariable(const daeDataReporterVariableValue* pVariableValue)
    {
        //if(value->m_strName != "")
        //    return;

        std::string strVariableName = pVariableValue->m_strName;
        boost::filesystem::path vtkFilename((boost::format("%05d.%s(t=%f).vtk") % m_outputCounter
                                                                                % strVariableName
                                                                                % m_dCurrentTime).str());
        boost::filesystem::path vtkPath(m_strOutputDirectory);
        vtkPath /= vtkFilename;

        m_callback(vtkPath.c_str(), strVariableName, pVariableValue->m_pValues, pVariableValue->m_nNumberOfPoints);
        m_outputCounter++;

        return true;
    }

public:
    real_t              m_dCurrentTime;
    int                 m_outputCounter;
    fnProcessSolution   m_callback;
    std::string         m_strOutputDirectory;
};


}
}

#endif
