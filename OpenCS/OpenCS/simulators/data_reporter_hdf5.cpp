/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License version 3 as published by the Free Software
Foundation. OpenCS is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with
the OpenCS software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "data_reporter_hdf5.h"
#include "cs_data_reporters.h"
#include <fstream>
#include <iomanip>

namespace cs
{
std::shared_ptr<csDataReporter_t> createDataReporter_HDF5(const std::string& hdf5FileName_res,
                                                          const std::string& hdf5FileName_der)
{
    return std::shared_ptr<csDataReporter_t>(new csDataReporter_hdf5(hdf5FileName_res,
                                                                     hdf5FileName_der));
}

csDataReporter_hdf5::csDataReporter_hdf5(const std::string& hdf5FileName_res,
                                         const std::string& hdf5FileName_der)
{
    m_currentTime            = 0.0;
    reportResults            = false;
    reportDerivatives        = false;
    m_hdf5FileNameResults    = hdf5FileName_res;
    m_hdf5FileNameDervatives = hdf5FileName_der;

    if(sizeof(real_t) == sizeof(double))
        datatypeFloat = H5T_NATIVE_DOUBLE;
    else if(sizeof(real_t) == sizeof(double))
        datatypeFloat = H5T_NATIVE_FLOAT;
}

csDataReporter_hdf5::~csDataReporter_hdf5()
{
    EndOfData();
    Disconnect();
}

std::string	csDataReporter_hdf5::GetName() const
{
    return "HDF5";
}

bool csDataReporter_hdf5::Connect(int rank)
{
    pe_rank = rank;

    reportResults = false;
    reportDerivatives = false;

    if(!m_hdf5FileNameResults.empty())
    {
        fileResults = H5Fcreate(m_hdf5FileNameResults.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if(fileResults >= 0)
            reportResults = true;
    }

    if(!m_hdf5FileNameDervatives.empty())
    {
        fileDerivatives = H5Fcreate(m_hdf5FileNameDervatives.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if(fileDerivatives >= 0)
            reportDerivatives = true;
    }

    return true;
}

bool csDataReporter_hdf5::IsConnected()
{
    return true;
}

bool csDataReporter_hdf5::Disconnect()
{
    herr_t status;

    if(reportResults)
        status = H5Fclose(fileResults);
    if(reportDerivatives)
        status = H5Fclose(fileDerivatives);

    reportResults     = false;
    reportDerivatives = false;

    return true;
}

bool csDataReporter_hdf5::RegisterVariables(const std::vector<std::string>& variableNames)
{
    size_t n = variableNames.size();
    std::vector<const char*> szarrVariableNames(n);
    for (size_t i = 0; i < n; i++)
        szarrVariableNames[i] = variableNames[i].c_str();

    std::string dset_name = "/variables";
    if(reportResults)
    {
        hsize_t dims[1];
        dims[0] = variableNames.size();
        hid_t dspace = H5Screate_simple(1, dims, NULL);

        hid_t datatype = H5Tcopy(H5T_C_S1);
        H5Tset_size (datatype, H5T_VARIABLE);

        hid_t dset = H5Dcreate2(fileResults, dset_name.c_str(),  datatype, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(dset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, szarrVariableNames.data());

        /* Flush the dataset to disk. */
        status = H5Fflush(dset, H5F_SCOPE_LOCAL);

        status = H5Dclose(dset);
        status = H5Sclose(dspace);
        status = H5Tclose(datatype);
    }
    if(reportDerivatives)
    {
        hsize_t dims[1];
        dims[0] = variableNames.size();
        hid_t dspace = H5Screate_simple(1, dims, NULL);

        hid_t datatype = H5Tcopy(H5T_C_S1);
        H5Tset_size (datatype, H5T_VARIABLE);

        hid_t dset = H5Dcreate2(fileDerivatives, dset_name.c_str(),  datatype, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(dset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, szarrVariableNames.data());

        /* Flush the dataset to disk. */
        status = H5Fflush(dset, H5F_SCOPE_LOCAL);

        status = H5Dclose(dset);
        status = H5Sclose(dspace);
        status = H5Tclose(datatype);
    }

    return true;
}

bool csDataReporter_hdf5::StartNewResultSet(real_t time)
{
    m_currentTime = time;
    m_times.push_back(time);
    return true;
}

bool csDataReporter_hdf5::EndOfData()
{
    if(m_times.size() == 0)
        return true;

    if(reportResults)
    {
        hsize_t dims[1];
        dims[0] = m_times.size();
        hid_t dspace = H5Screate_simple(1, dims, NULL);

        std::string dset_name = "/times";
        hid_t dset = H5Dcreate2(fileResults, dset_name.c_str(),  datatypeFloat, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(dset, datatypeFloat, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_times[0]);

        status = H5Dclose(dset);
        status = H5Sclose(dspace);

        /* Flush the hdf5 file to disk. */
        status = H5Fflush(fileResults, H5F_SCOPE_LOCAL);
    }
    if(reportDerivatives)
    {
        hsize_t dims[1];
        dims[0] = m_times.size();
        hid_t dspace = H5Screate_simple(1, dims, NULL);

        std::string dset_name = "/times";
        hid_t dset = H5Dcreate2(fileDerivatives, dset_name.c_str(),  datatypeFloat, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = H5Dwrite(dset, datatypeFloat, H5S_ALL, H5S_ALL, H5P_DEFAULT, &m_times[0]);

        status = H5Dclose(dset);
        status = H5Sclose(dspace);

        /* Flush the hdf5 file to disk. */
        status = H5Fflush(fileDerivatives, H5F_SCOPE_LOCAL);
    }

    m_times.clear();

    return true;
}

bool csDataReporter_hdf5::SendVariables(const real_t* values, const size_t Nequations)
{
    static int counter = 0;

    if(!reportResults)
        return true;

    hsize_t dims[1];
    dims[0] = Nequations;
    hid_t dspace   = H5Screate_simple(1, dims, NULL);
    dims[0] = 1;
    hid_t dspace_t = H5Screate_simple(1, dims, NULL);

    std::string dset_name = "/" + std::to_string(counter);
    hid_t dset = H5Dcreate2(fileResults, dset_name.c_str(),  datatypeFloat, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(dset, datatypeFloat, H5S_ALL, H5S_ALL, H5P_DEFAULT, values);

    /* Create a dataset attribute (time). */
    hid_t attribute_id = H5Acreate2 (dset, "time", datatypeFloat, dspace_t, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attribute_id, datatypeFloat, &m_currentTime);
    status = H5Aclose(attribute_id);

    /* Flush the dataset to disk. */
    status = H5Fflush(dset, H5F_SCOPE_LOCAL);

    status = H5Sclose(dspace);
    status = H5Sclose(dspace_t);
    status = H5Dclose(dset);

    counter++;

    return true;
}

bool csDataReporter_hdf5::SendDerivatives(const real_t* derivatives, const size_t Nequations)
{
    static int counter = 0;

    if(!reportDerivatives)
        return true;

    hsize_t dims[1];
    dims[0] = Nequations;
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    dims[0] = 1;
    hid_t dspace_t = H5Screate_simple(1, dims, NULL);

    std::string dset_name = "/" + std::to_string(counter);
    hid_t dset = H5Dcreate2(fileDerivatives, dset_name.c_str(), datatypeFloat, dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(dset, datatypeFloat, H5S_ALL, H5S_ALL, H5P_DEFAULT, derivatives);

    /* Create a dataset attribute (time). */
    hid_t attribute_id = H5Acreate2 (dset, "time", datatypeFloat, dspace_t, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Awrite(attribute_id, datatypeFloat, &m_currentTime);
    status = H5Aclose(attribute_id);

    /* Flush the dataset to disk. */
    status = H5Fflush(dset, H5F_SCOPE_LOCAL);

    status = H5Sclose(dspace);
    status = H5Sclose(dspace_t);
    status = H5Dclose(dset);

    counter++;

    return true;
}

}
