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
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "../cs_model.h"
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;

namespace cs
{
static std::string formatFilename(const std::string& inputDirectory, const std::string& inputFile, int pe_rank)
{
    const size_t bsize = 1024;
    char filename[bsize];

    /* Format file name. */
    snprintf(filename, bsize, csModel_t::inputFileNameTemplate, inputFile.c_str(), pe_rank);

    /* Compose the file path.
     * It is assumed that the inputDirectory is full path or relative path to the current diectory. */
    filesystem::path inputDirectoryPath = filesystem::absolute( filesystem::path(inputDirectory) );
    std::string filePath = (inputDirectoryPath / filename).string();

    return filePath;
}

static void loadModelStructure(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = formatFilename(inputDirectory, csModel_t::modelStructureFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    int32_t fileType;
    int32_t opencsVersion;
    f.read((char*)&fileType,       sizeof(int32_t));
    f.read((char*)&opencsVersion,  sizeof(int32_t));
    if(fileType != eInputFile_ModelStructure)
        throw std::runtime_error("File: " + filePath + " does not contain model variables");

    f.read((char*)&model->structure.Nequations_total,   sizeof(uint32_t));
    f.read((char*)&model->structure.Nequations,         sizeof(uint32_t));
    f.read((char*)&model->structure.Ndofs,              sizeof(uint32_t));
    f.read((char*)&model->structure.isODESystem,        sizeof(bool));

    int32_t Nitems;

    // DOF values
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->structure.dofValues.resize(Nitems);
        f.read((char*)(&model->structure.dofValues[0]), sizeof(real_t) * Nitems);
    }

    // Variable values
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->structure.variableValues.resize(Nitems);
        f.read((char*)(&model->structure.variableValues[0]), sizeof(real_t) * Nitems);
    }

    // Variable derivatives
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->structure.variableDerivatives.resize(Nitems);
        f.read((char*)(&model->structure.variableDerivatives[0]), sizeof(real_t) * Nitems);
    }

    // Absolute tolerances
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->structure.absoluteTolerances.resize(Nitems);
        f.read((char*)(&model->structure.absoluteTolerances[0]), sizeof(real_t) * Nitems);
    }

    // Variable types
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->structure.variableTypes.resize(Nitems);
        f.read((char*)(&model->structure.variableTypes[0]), sizeof(int32_t) * Nitems);
    }

    // variable_names (skipped)
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems == 0)
    {
        model->structure.variableNames.resize(model->structure.Nequations);
        const size_t vnsize = 1024;
        char varname[vnsize];

        /* Format file name. */
        for(int i = 0; i < model->structure.Nequations; i++)
        {
            snprintf(varname, vnsize, "y(%d)", i);
            model->structure.variableNames[i] = varname;
        }
    }
    else
    {
        model->structure.variableNames.resize(Nitems);

        int32_t length;
        char name[4096];
        for(int i = 0; i < Nitems; i++)
        {
            // name length
            f.read((char*)&length,  sizeof(int32_t));

            // name contents
            f.read((char*)(&name[0]), sizeof(char) * length);
            name[length] = '\0';

            model->structure.variableNames[i] = std::string(name);
        }
    }

    f.close();
}

static void saveModelStructure(csModel_t* model, const std::string& outputDirectory)
{
    std::ofstream f;
    std::string filePath = formatFilename(outputDirectory, csModel_t::modelStructureFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    /* Write header. */
    int32_t fileType      = eInputFile_ModelStructure;
    int32_t opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    f.write((char*)&model->structure.Nequations_total,   sizeof(uint32_t));
    f.write((char*)&model->structure.Nequations,         sizeof(uint32_t));
    f.write((char*)&model->structure.Ndofs,              sizeof(uint32_t));
    f.write((char*)&model->structure.isODESystem,        sizeof(bool));

    int32_t Nitems;

    // DOF values
    Nitems = model->structure.dofValues.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        f.write((char*)(&model->structure.dofValues[0]), sizeof(real_t) * Nitems);
    }

    // Variable values
    Nitems = model->structure.variableValues.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        f.write((char*)(&model->structure.variableValues[0]), sizeof(real_t) * Nitems);
    }

    // Variable derivatives
    Nitems = model->structure.variableDerivatives.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        f.write((char*)(&model->structure.variableDerivatives[0]), sizeof(real_t) * Nitems);
    }

    // Absolute tolerances
    Nitems = model->structure.absoluteTolerances.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        f.write((char*)(&model->structure.absoluteTolerances[0]), sizeof(real_t) * Nitems);
    }

    // Variable types
    Nitems = model->structure.variableTypes.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        f.write((char*)(&model->structure.variableTypes[0]), sizeof(int32_t) * Nitems);
    }

    // variable_names (skipped)
    Nitems = model->structure.variableNames.size();
    f.write((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        for(int i = 0; i < Nitems; i++)
        {
            std::string name = model->structure.variableNames[i];
            int32_t length = name.size();

            // name length
            f.write((char*)&length,  sizeof(int32_t));

            // name contents
            f.write((char*)(&name[0]), sizeof(char) * length);
        }
    }

    f.close();
}

static void loadPartitionData(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = formatFilename(inputDirectory, csModel_t::partitionDataFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    int32_t fileType;
    int32_t opencsVersion;
    f.read((char*)&fileType,       sizeof(int32_t));
    f.read((char*)&opencsVersion,  sizeof(int32_t));
    if(fileType != eInputFile_PartitionData)
        throw std::runtime_error("File: " + filePath + " does not contain partition data");

    // MPI functions require integers so everything is saved as singed integers
    if(sizeof(int32_t) != sizeof(int))
        throw std::runtime_error("Invalid size of int (must be 4 bytes)");

    // foreign_indexes
    int32_t Nforeign;
    f.read((char*)&Nforeign,  sizeof(int32_t));
    if(Nforeign > 0)
    {
        model->partitionData.foreignIndexes.resize(Nforeign);
        f.read((char*)(&model->partitionData.foreignIndexes[0]), sizeof(int32_t) * Nforeign);
    }

    // bi_to_bi_local
    int32_t Nbi_to_bi_local_pairs, bi, bi_local;
    f.read((char*)&Nbi_to_bi_local_pairs,  sizeof(int32_t));
    //model->g_partitionData.bi_to_bi_local.reserve(Nbi_to_bi_local_pairs);
    for(int32_t i = 0; i < Nbi_to_bi_local_pairs; i++)
    {
        f.read((char*)&bi,       sizeof(int32_t));
        f.read((char*)&bi_local, sizeof(int32_t));
        model->partitionData.biToBiLocal[bi] = bi_local;
    }

    // sendToIndexes
    int32_t Nsend_to;
    f.read((char*)&Nsend_to,  sizeof(int32_t));
    for(int32_t i = 0; i < Nsend_to; i++)
    {
        int32_t rank, Nindexes;
        std::vector<int32_t> indexes;

        f.read((char*)&rank,     sizeof(int32_t));
        f.read((char*)&Nindexes, sizeof(int32_t));
        indexes.resize(Nindexes);
        f.read((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
        model->partitionData.sendToIndexes[rank] = indexes;

        if(!std::is_sorted(indexes.begin(), indexes.end()))
            throw std::runtime_error("sendToIndexes indexes are not sorted");
    }

    // receiveFromIndexes
    int32_t Nreceive_from;
    f.read((char*)&Nreceive_from,  sizeof(int32_t));
    for(int32_t i = 0; i < Nreceive_from; i++)
    {
        int32_t rank, Nindexes;
        std::vector<int32_t> indexes;

        f.read((char*)&rank,     sizeof(int32_t));
        f.read((char*)&Nindexes, sizeof(int32_t));
        indexes.resize(Nindexes);
        f.read((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
        model->partitionData.receiveFromIndexes[rank] = indexes;

        if(!std::is_sorted(indexes.begin(), indexes.end()))
            throw std::runtime_error("receiveFromIndexes indexes are not sorted");
    }

    f.close();
}

static void savePartitionData(csModel_t* model, const std::string& outputDirectory)
{
    std::ofstream f;
    std::string filePath = formatFilename(outputDirectory, csModel_t::partitionDataFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    /* Write header. */
    int32_t fileType      = eInputFile_PartitionData;
    int32_t opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    // MPI functions require integers so everything is saved as singed integers
    if(sizeof(int32_t) != sizeof(int))
        throw std::runtime_error("Invalid size of int (must be 4 bytes)");

    // foreign_indexes
    int32_t Nforeign = model->partitionData.foreignIndexes.size();
    f.write((char*)&Nforeign,  sizeof(int32_t));
    if(Nforeign > 0)
    {
        f.write((char*)(&model->partitionData.foreignIndexes[0]), sizeof(int32_t) * Nforeign);
    }

    // bi_to_bi_local
    int32_t Nbi_to_bi_local_pairs, bi, bi_local;
    Nbi_to_bi_local_pairs = model->partitionData.biToBiLocal.size();
    f.write((char*)&Nbi_to_bi_local_pairs,  sizeof(int32_t));
    for(std::map<int32_t,int32_t>::const_iterator it = model->partitionData.biToBiLocal.begin(); it != model->partitionData.biToBiLocal.end(); it++)
    {
        bi       = it->first;
        bi_local = it->second;
        f.write((char*)&bi,       sizeof(int32_t));
        f.write((char*)&bi_local, sizeof(int32_t));
    }

    // sendToIndexes
    int32_t Nsend_to = model->partitionData.sendToIndexes.size();
    f.write((char*)&Nsend_to,  sizeof(int32_t));
    for(csPartitionIndexMap::const_iterator it = model->partitionData.sendToIndexes.begin(); it != model->partitionData.sendToIndexes.end(); it++)
    {
        int32_t              rank     = it->first;
        std::vector<int32_t> indexes  = it->second;
        int32_t              Nindexes = indexes.size();

        std::sort(indexes.begin(), indexes.end());

        f.write((char*)&rank,         sizeof(int32_t));
        f.write((char*)&Nindexes,     sizeof(int32_t));
        f.write((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
    }

    // receiveFromIndexes
    int32_t Nreceive_from = model->partitionData.receiveFromIndexes.size();
    f.write((char*)&Nreceive_from,  sizeof(int32_t));
    for(csPartitionIndexMap::const_iterator it = model->partitionData.receiveFromIndexes.begin(); it != model->partitionData.receiveFromIndexes.end(); it++)
    {
        int32_t              rank     = it->first;
        std::vector<int32_t> indexes  = it->second;
        int32_t              Nindexes = indexes.size();

        std::sort(indexes.begin(), indexes.end());

        f.write((char*)&rank,         sizeof(int32_t));
        f.write((char*)&Nindexes,     sizeof(int32_t));
        f.write((char*)(&indexes[0]), sizeof(int32_t) * Nindexes);
    }

    f.close();
}

static void loadModelEquations(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = formatFilename(inputDirectory, csModel_t::modelEquationsFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    int32_t fileType;
    int32_t opencsVersion;
    f.read((char*)&fileType,       sizeof(int32_t));
    f.read((char*)&opencsVersion,  sizeof(int32_t));
    if(fileType != eInputFile_ModelEquations)
        throw std::runtime_error("File: " + filePath + " does not contain model equations");

    uint32_t Ncs  = 0;
    uint32_t Nasi = 0;

    f.read((char*)&Nasi, sizeof(uint32_t));
    f.read((char*)&Ncs,  sizeof(uint32_t));

    model->equations.activeEquationSetIndexes.resize(Nasi);
    model->equations.computeStacks.resize(Ncs);

    f.read((char*)(&model->equations.activeEquationSetIndexes[0]), sizeof(uint32_t)             * Nasi);
    f.read((char*)(&model->equations.computeStacks[0]),            sizeof(csComputeStackItem_t) * Ncs);

    f.close();
}

static void saveModelEquations(csModel_t* model, const std::string& outputDirectory)
{
    std::ofstream f;
    std::string filePath = formatFilename(outputDirectory, csModel_t::modelEquationsFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    /* Write header. */
    int32_t fileType      = eInputFile_ModelEquations;
    int32_t opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    uint32_t Ncs  = model->equations.computeStacks.size();
    uint32_t Nasi = model->equations.activeEquationSetIndexes.size();

    f.write((char*)&Nasi, sizeof(uint32_t));
    f.write((char*)&Ncs,  sizeof(uint32_t));

    f.write((char*)(&model->equations.activeEquationSetIndexes[0]), sizeof(uint32_t)             * Nasi);
    f.write((char*)(&model->equations.computeStacks[0]),            sizeof(csComputeStackItem_t) * Ncs);

    f.close();
}

static void loadIncidenceMatrix(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = formatFilename(inputDirectory, csModel_t::sparsityPatternFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    int32_t fileType;
    int32_t opencsVersion;
    f.read((char*)&fileType,       sizeof(int32_t));
    f.read((char*)&opencsVersion,  sizeof(int32_t));
    if(fileType != eInputFile_SparsityPattern)
        throw std::runtime_error("File: " + filePath + " does not contain incidence matrix");

    uint32_t Nequations = 0;
    uint32_t Nji        = 0;
    f.read((char*)&Nequations, sizeof(uint32_t));
    f.read((char*)&Nji,        sizeof(uint32_t));

    model->sparsityPattern.Nequations = Nequations;
    model->sparsityPattern.Nnz        = Nji;

    if(Nequations > 0)
    {
        model->sparsityPattern.rowIndexes.resize(Nequations+1);
        f.read((char*)(&model->sparsityPattern.rowIndexes[0]), sizeof(uint32_t) * (Nequations+1));
    }
    if(Nji > 0)
    {
        model->sparsityPattern.incidenceMatrixItems.resize(Nji);
        f.read((char*)(&model->sparsityPattern.incidenceMatrixItems[0]), sizeof(csIncidenceMatrixItem_t) * Nji);
    }

    f.close();
}

static void saveIncidenceMatrix(csModel_t* model, const std::string& outputDirectory)
{
    std::ofstream f;
    std::string filePath = formatFilename(outputDirectory, csModel_t::sparsityPatternFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    /* Write header. */
    int32_t fileType      = eInputFile_SparsityPattern;
    int32_t opencsVersion = OPENCS_VERSION;
    f.write((char*)&fileType,       sizeof(int32_t));
    f.write((char*)&opencsVersion,  sizeof(int32_t));

    uint32_t Nequations = model->sparsityPattern.rowIndexes.size() - 1;
    uint32_t Nji        = model->sparsityPattern.incidenceMatrixItems.size();

    if(model->sparsityPattern.Nequations != Nequations)
        throw std::runtime_error("Invalid number of equations");
    if(model->sparsityPattern.Nnz != Nji)
        throw std::runtime_error("Invalid number of non-zero items");

    f.write((char*)&Nequations, sizeof(uint32_t));
    f.write((char*)&Nji,        sizeof(uint32_t));

    if(Nequations > 0)
    {
        f.write((char*)(&model->sparsityPattern.rowIndexes[0]), sizeof(uint32_t) * (Nequations+1));
    }
    if(Nji > 0)
    {
        f.write((char*)(&model->sparsityPattern.incidenceMatrixItems[0]), sizeof(csIncidenceMatrixItem_t) * Nji);
    }

    f.close();
}

csModel_t::~csModel_t()
{
}

csModel_t::csModel_t()
{
    pe_rank  = 0;
}

void csModel_t::LoadModel(const std::string& inputDirectory)
{
    /* Load model structure. */
    loadModelStructure(this, inputDirectory);

    /* Load partition data. */
    loadPartitionData(this, inputDirectory);

    /* Load model equations. */
    loadModelEquations(this, inputDirectory);

    /* Load incidence matrix. */
    loadIncidenceMatrix(this, inputDirectory);
}

void csModel_t::SaveModel(const std::string& outputDirectory)
{
    /* Create directories, if missing. */
    filesystem::path outputDirectoryPath = filesystem::absolute( filesystem::path(outputDirectory) );
    if(!filesystem::is_directory(outputDirectoryPath))
        filesystem::create_directories(outputDirectoryPath);

    /* Save model structure. */
    saveModelStructure(this, outputDirectory);

    /* Save partition data. */
    savePartitionData(this, outputDirectory);

    /* Save model equations. */
    saveModelEquations(this, outputDirectory);

    /* Save incidence matrix. */
    saveIncidenceMatrix(this, outputDirectory);
}

}
