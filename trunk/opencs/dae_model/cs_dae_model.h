/***********************************************************************************
*                 OpenCS Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
OpenCS is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#ifndef CS_DAE_MODEL_IMPLEMENTATION_H
#define CS_DAE_MODEL_IMPLEMENTATION_H

#include <string>
#include <vector>
#include <map>
#include "../opencs.h"

namespace cs
{
typedef std::map< int32_t, std::pair< std::vector<real_t>, std::vector<real_t> > >  mpiSyncValuesMap;
struct mpiValuesData
{
    mpiSyncValuesMap  sendToIndexes;
    mpiSyncValuesMap  receiveFromIndexes;
};

typedef std::map< int32_t, std::pair< std::vector<real_t*>,std::vector<real_t*> > > mpiSyncPointersMap;
struct mpiPointersData
{
    mpiSyncPointersMap sendToIndexes;
    mpiSyncPointersMap receiveFromIndexes;
};

class csDAEModelImplementation_t : public csDAEModel_t
{
public:
    csDAEModelImplementation_t();
    virtual ~csDAEModelImplementation_t();

    void Load(const std::string& inputDirectory, csComputeStackEvaluator_t* csEvaluator);
    void Free();
    void GetDAESystemStructure(int& N,
                               int& NNZ,
                               std::vector<int>& IA,
                               std::vector<int>& JA);
    void EvaluateResiduals(real_t time, real_t* residuals);
    void EvaluateJacobian(real_t time, real_t inverseTimeStep, csMatrixAccess_t* ma);
    void SetAndSynchroniseData(real_t  time, real_t* values, real_t* time_derivatives);
    int  NumberOfRoots();
    void Roots(real_t time, real_t* values, real_t* time_derivatives, real_t* roots);
    bool CheckForDiscontinuities(real_t time, real_t* values, real_t* time_derivatives);
    csDiscontinuityType ExecuteActions(real_t time, real_t* values, real_t* time_derivatives);

protected:
    void InitializeValuesReferences();
    void CheckSynchronisationIndexes();

protected:
    /* Current time in model, set by SetAndSynchroniseData function. */
    real_t                  currentTime;

    /* Internal storage containers for evaluation of equations. */
    std::vector<real_t>     m_values;
    std::vector<real_t>     m_timeDerivatives;
    std::vector<real_t>     m_dofs;
    std::vector<real_t>     m_jacobian;

    /* Data related to MPI communication (include index mappings for both owned and foreign indexes). */
    std::map<int, real_t*>  m_mapValues;
    std::map<int, real_t*>  m_mapTimeDerivatives;

    /* ComputeStack-related data. */
    uint32_t*               m_activeEquationSetIndexes;
    csComputeStackItem_t*   m_computeStacks;
    csJacobianMatrixItem_t* m_jacobianMatrixItems;
    uint32_t                m_numberOfJacobianItems;
    uint32_t                m_numberOfComputeStackItems;

    /* Internal map of values/derivatives arrays. */
    mpiValuesData           m_mapValuesData;
    mpiPointersData         m_mapPointersData;
};

}

#endif
