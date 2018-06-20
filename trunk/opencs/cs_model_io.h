/***********************************************************************************
                 OpenCS Project: www.daetools.com
                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_MODEL_IO_H
#define CS_MODEL_IO_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "cs_model.h"

namespace cs
{
const char* inputFileNameTemplate  = "%s-%05d.bin";
const char* modelEquationsFileName = "model_equations";
const char* modelVariablesFileName = "model_variables";
const char* partitionDataFileName  = "partition_data";
const char* jacobianDataFileName   = "jacobian_data";

static std::string getFullPathForPE(const std::string& inputDirectory, const std::string& inputFile, int pe_rank)
{
    const size_t bsize = 1024;
    char filename[bsize];

    /* Format file name. */
    snprintf(filename, bsize, inputFileNameTemplate, inputFile.c_str(), pe_rank);

    /* Compose the file path.
     * It is assumed that the inputDirectory is full path or relative path to the current diectory. */
    std::string filePath = inputDirectory + "/" + filename;

    return filePath;
}

static void loadModelVariables(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = getFullPathForPE(inputDirectory, modelVariablesFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    f.read((char*)&model->Nequations,         sizeof(uint32_t));
    f.read((char*)&model->Nequations_local,   sizeof(uint32_t));
    f.read((char*)&model->Ndofs,              sizeof(uint32_t));
    f.read((char*)&model->quasiSteadyState,   sizeof(bool));

    int32_t Nitems;

    // dofs
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->dofs.resize(Nitems);
        f.read((char*)(&model->dofs[0]), sizeof(real_t) * Nitems);
    }

    // init_values
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->variableValues.resize(Nitems);
        f.read((char*)(&model->variableValues[0]), sizeof(real_t) * Nitems);
    }

    // init_derivatives
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->variableDerivatives.resize(Nitems);
        f.read((char*)(&model->variableDerivatives[0]), sizeof(real_t) * Nitems);
    }

    // absolute_tolerances
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->absoluteTolerances.resize(Nitems);
        f.read((char*)(&model->absoluteTolerances[0]), sizeof(real_t) * Nitems);
    }

    // ids
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems > 0)
    {
        model->ids.resize(Nitems);
        f.read((char*)(&model->ids[0]), sizeof(int32_t) * Nitems);
    }

    // variable_names (skipped)
    f.read((char*)&Nitems,  sizeof(int32_t));
    if(Nitems == 0)
    {
        model->variableNames.resize(model->Nequations_local);
        const size_t vnsize = 1024;
        char varname[vnsize];

        /* Format file name. */
        for(int i = 0; i < model->Nequations_local; i++)
        {
            snprintf(varname, vnsize, "y(%d)", i);
            model->variableNames[i] = varname;
        }
    }
    else
    {
        model->variableNames.resize(Nitems);

        int32_t length;
        char name[4096];
        for(int i = 0; i < Nitems; i++)
        {
            // Read string length
            f.read((char*)&length,  sizeof(int32_t));

            // Read string
            f.read((char*)(&name[0]), sizeof(char) * length);
            name[length] = '\0';

            model->variableNames[i] = std::string(name);
        }
    }

    f.close();
}

static void loadPartitionData(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = getFullPathForPE(inputDirectory, partitionDataFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

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

static void loadModelEquations(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = getFullPathForPE(inputDirectory, modelEquationsFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    uint32_t Ncs  = 0;
    uint32_t Nasi = 0;

    f.read((char*)&Nasi, sizeof(uint32_t));
    f.read((char*)&Ncs,  sizeof(uint32_t));

    model->activeEquationSetIndexes.resize(Nasi);
    model->computeStacks.resize(Ncs);

    f.read((char*)(&model->activeEquationSetIndexes[0]), sizeof(uint32_t)             * Nasi);
    f.read((char*)(&model->computeStacks[0]),            sizeof(csComputeStackItem_t) * Ncs);
    f.close();
}

static void loadJacobianData(csModel_t* model, const std::string& inputDirectory)
{
    std::ifstream f;
    std::string filePath = getFullPathForPE(inputDirectory, jacobianDataFileName, model->pe_rank);
    f.open(filePath, std::ios_base::binary);
    if(!f.is_open())
        throw std::runtime_error("Cannot open " + filePath + " file");

    uint32_t Nji = 0;

    f.read((char*)&Nji, sizeof(uint32_t));

    model->jacobianMatrixItems.resize(Nji);

    f.read((char*)(&model->jacobianMatrixItems[0]), sizeof(csJacobianMatrixItem_t) * Nji);
    f.close();
}

static void csLoadModel(csModel_t* model, const std::string& inputDirectory)
{
    /* Load model structure. */
    loadModelVariables(model, inputDirectory);

    /* Load partition data. */
    loadPartitionData(model, inputDirectory);

    /* Load model equations. */
    loadModelEquations(model, inputDirectory);

    /* Load Jacobian data (inidence matrix). */
    loadJacobianData(model, inputDirectory);
}

}
#endif
