#include "fe_solvers.h"
using namespace dae::fe_solver::convection_diffusion_dealii;

daeModel* CreateModel(const std::string& modelClass, const std::string& name, daeModel* parent, const std::string& description, const std::map<std::string, double>& modelOptions)
{
//    if(modelClass == "daeConvectionDiffusion_1D")
//        return new daeConvectionDiffusion_1D(name, parent, description);
//    else if(modelClass == "daeConvectionDiffusion_2D")
//        return new daeConvectionDiffusion_2D(name, parent, description);
//    else if(modelClass == "daeConvectionDiffusion_3D")
//        return new daeConvectionDiffusion_3D(name, parent, description);
//    else
        return NULL;
}
