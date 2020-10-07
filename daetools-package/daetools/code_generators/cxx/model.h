#ifndef CV_4_MODEL_H
#define CV_4_MODEL_H

#include "adouble.h"
#include <vector>
#include <map>

using adolc::sin_;
using adolc::cos_;
using adolc::adouble;

class cv_4
{
public:
    typedef adouble (cv_4:: *ad_function_ptr)(int, int);

    adouble u0;
    adouble v0;
    adouble w0;
    adouble ni;
    adouble eps;
    adouble t;
    real_t  inverseTimeStep;

    int Nx;
    int Ny;
    adouble dx;
    adouble dy;

    // Variable values
    real_t* u_data;
    real_t* v_data;
    real_t* uman_data;
    real_t* vman_data;

    // Variable derivatives
    real_t* du_dt_data;
    real_t* dv_dt_data;
    real_t* duman_dt_data;
    real_t* dvman_dt_data;

    std::vector<adouble> x_domain;
    std::vector<adouble> y_domain;

    int    jacobianIndex;

    int u_start_index;
    int v_start_index;
    int uman_start_index;
    int vman_start_index;

    std::map< int, std::vector<int> > jacobianMatrixIndexes;

    cv_4()
    {
        u0  = adouble(1.0);
        v0  = adouble(1.0);
        w0  = adouble(0.1);
        ni  = adouble(0.7);
        eps = adouble(0.001);

        Nx  = 120 + 1;
        Ny  = 96 + 1;
        dx  = adouble((0.7 + 0.1) / (Nx-1));
        dy  = adouble((0.8 - 0.2) / (Ny-1));

        u_start_index    = 0*Nx*Ny;
        v_start_index    = 1*Nx*Ny;
        uman_start_index = 2*Nx*Ny;
        vman_start_index = 3*Nx*Ny;

        x_domain.resize(Nx);
        y_domain.resize(Ny);
        for(int x = 0; x < Nx; x++)
            x_domain[x] = -0.1 + x * dx;
        for(int y = 0; y < Ny; y++)
            y_domain[y] = 0.2 + y * dy;

        u_data         = NULL;
        v_data         = NULL;
        du_dt_data     = NULL;
        dv_dt_data     = NULL;
        uman_data      = NULL;
        vman_data      = NULL;
        duman_dt_data  = NULL;
        dvman_dt_data  = NULL;
        duman_dt_data  = NULL;
        dvman_dt_data  = NULL;

        inverseTimeStep = 0.0;
        jacobianIndex   = -1;

        /* jacobianMatrixIndexes map is populated from the ComputeStack jacobian data. */
    }

    adouble df_dx   (ad_function_ptr f, int x, int y)
    {
        return ( ((this->*f)(x+1,  y) - (this->*f)(x-1,  y)) / (2*dx) );
    }

    adouble df_dy   (ad_function_ptr f, int x, int y)
    {
        return ( ((this->*f)(x,  y+1) - (this->*f)(x,  y-1)) / (2*dy) );
    }

    adouble d2f_dx2 (ad_function_ptr f, int x, int y)
    {
        return ( ((this->*f)(x+1,  y) - 2*(this->*f)(x,y) + (this->*f)(x-1,  y)) / (dx*dx) );
    }

    adouble d2f_dy2 (ad_function_ptr f, int x, int y)
    {
        return ( ((this->*f)(x,  y+1) - 2*(this->*f)(x,y) + (this->*f)(x,  y-1)) / (dy*dy) );
    }

    adouble product_rule_dx (ad_function_ptr f1, ad_function_ptr f2, int x, int y)
    {
        return ( df_dx(f1,x,y)*(this->*f2)(x,y) + (this->*f1)(x,y)*df_dx(f2,x,y) );
    }

    adouble product_rule_dy (ad_function_ptr f1, ad_function_ptr f2, int x, int y)
    {
        return ( df_dy(f1,x,y)*(this->*f2)(x,y) + (this->*f1)(x,y)*df_dy(f2,x,y) );
    }

    int getIndex(int x, int y)
    {
        if(x < 0 || x >= Nx)
            daeThrowException("Invalid x index");
        if(y < 0 || y >= Ny)
            daeThrowException("Invalid y index");
        return Ny*x + y;
    }

    // Data access functions
    adouble    u       (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( u_data[index], (jacobianIndex == u_start_index+index ? 1.0 : 0.0) );
    }
    adouble    v       (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( v_data[index], (jacobianIndex == v_start_index+index ? 1.0 : 0.0) );
    }
    adouble    du_dt   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( du_dt_data[index], (jacobianIndex == u_start_index+index ? inverseTimeStep : 0.0) );
    }
    adouble    dv_dt   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( dv_dt_data[index], (jacobianIndex == v_start_index+index ? inverseTimeStep : 0.0) );
    }

    adouble    u_man   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( uman_data[index], (jacobianIndex == uman_start_index+index ? 1.0 : 0.0) );
    }
    adouble    v_man   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( vman_data[index], (jacobianIndex == vman_start_index+index ? 1.0 : 0.0) );
    }
    adouble    duman_dt   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( duman_dt_data[index], (jacobianIndex == uman_start_index+index ? inverseTimeStep : 0.0) );
    }
    adouble    dvman_dt   (int x, int y)
    {
        int index = getIndex(x,y);
        return adouble( dvman_dt_data[index], (jacobianIndex == vman_start_index+index ? inverseTimeStep : 0.0) );
    }

    // Domain functions
    adouble    xd      (int x)        { return x_domain[x]; }
    adouble    yd      (int y)        { return y_domain[y]; }

    adouble    duu_dx  (int x, int y) { return product_rule_dx(&cv_4::u, &cv_4::u, x,y); }
    adouble    duv_dy  (int x, int y) { return product_rule_dy(&cv_4::u, &cv_4::v, x,y); }
    adouble    d2u_dx2 (int x, int y) { return d2f_dx2(&cv_4::u, x,y); }
    adouble    d2u_dy2 (int x, int y) { return d2f_dy2(&cv_4::u, x,y); }

    adouble    dvu_dx  (int x, int y) { return product_rule_dx(&cv_4::v, &cv_4::u, x,y); }
    adouble    dvv_dy  (int x, int y) { return product_rule_dy(&cv_4::v, &cv_4::v, x,y); }
    adouble    d2v_dx2 (int x, int y) { return d2f_dx2(&cv_4::v, x,y); }
    adouble    d2v_dy2 (int x, int y) { return d2f_dy2(&cv_4::v, x,y); }

    adouble    dumum_dx (int x, int y) { return product_rule_dx(&cv_4::um, &cv_4::um, x,y); }
    adouble    dumvm_dy (int x, int y) { return product_rule_dy(&cv_4::um, &cv_4::vm, x,y); }
    adouble    d2um_dx2 (int x, int y) { return d2f_dx2(&cv_4::um, x,y); }
    adouble    d2um_dy2 (int x, int y) { return d2f_dy2(&cv_4::um, x,y); }

    adouble    dvmum_dx (int x, int y) { return product_rule_dx(&cv_4::vm, &cv_4::um, x,y); }
    adouble    dvmvm_dy (int x, int y) { return product_rule_dy(&cv_4::vm, &cv_4::vm, x,y); }
    adouble    d2vm_dx2 (int x, int y) { return d2f_dx2(&cv_4::vm, x,y); }
    adouble    d2vm_dy2 (int x, int y) { return d2f_dy2(&cv_4::vm, x,y); }

    // Manufactured solutions and source terms
    adouble    um       (int x, int y) { return u0 * (sin_(xd(x)*xd(x) + yd(y)*yd(y) + w0*t) + eps); }
    adouble    vm       (int x, int y) { return v0 * (cos_(xd(x)*xd(x) + yd(y)*yd(y) + w0*t) + eps); }

    adouble    Su       (int x, int y) { return duman_dt(x,y) + (dumum_dx(x,y) + dumvm_dy(x,y)) - ni * (d2um_dx2(x,y) + d2um_dy2(x,y)); }
    adouble    Sv       (int x, int y) { return dvman_dt(x,y) + (dvmum_dx(x,y) + dvmvm_dy(x,y)) - ni * (d2vm_dx2(x,y) + d2vm_dy2(x,y)); }

    void EvaluateResiduals(double time, double* values, double* derivs, double* residuals)
    {
        t           = adouble(time);

        u_data      = &values[u_start_index];
        v_data      = &values[v_start_index];
        uman_data   = &values[uman_start_index];
        vman_data   = &values[vman_start_index];

        du_dt_data    = &derivs[u_start_index];
        dv_dt_data    = &derivs[v_start_index];
        duman_dt_data = &derivs[uman_start_index];
        dvman_dt_data = &derivs[vman_start_index];

        int eq = 0;
        int x, y;

        // u component
        for(x = 1; x < Nx-1; x++)
        {
            for(y = 1; y < Ny-1; y++)
            {
                residuals[eq++] = (du_dt(x,y) + (duu_dx(x,y) + duv_dy(x,y)) - ni * (d2u_dx2(x,y) + d2u_dy2(x,y)) - Su(x,y)).getValue();
            }
        }
        // u BCs
        for(x = 1; x < Nx-1; x++)
        {
            y = 0;
            residuals[eq++] = (u(x,y) - um(x,y)).getValue();
        }
        for(x = 1; x < Nx-1; x++)
        {
            y = Ny-1;
            residuals[eq++] = (u(x,y) - um(x,y)).getValue();
        }
        for(y = 0; y < Ny; y++)
        {
            x = 0;
            residuals[eq++] = (u(x,y) - um(x,y)).getValue();
        }
        for(y = 0; y < Ny; y++)
        {
            x = Nx-1;
            residuals[eq++] = (u(x,y) - um(x,y)).getValue();
        }

        // v component
        for(x = 1; x < Nx-1; x++)
        {
            for(y = 1; y < Ny-1; y++)
            {
                residuals[eq++] = (dv_dt(x,y) + (dvu_dx(x,y) + dvv_dy(x,y)) - ni * (d2v_dx2(x,y) + d2v_dy2(x,y)) - Sv(x,y)).getValue();
            }
        }
        // v BCs
        for(x = 1; x < Nx-1; x++)
        {
            y = 0;
            residuals[eq++] = (v(x,y) - vm(x,y)).getValue();
        }
        for(x = 1; x < Nx-1; x++)
        {
            y = Ny-1;
            residuals[eq++] = (v(x,y) - vm(x,y)).getValue();
        }
        for(y = 0; y < Ny; y++)
        {
            x = 0;
            residuals[eq++] = (v(x,y) - vm(x,y)).getValue();
        }
        for(y = 0; y < Ny; y++)
        {
            x = Nx-1;
            residuals[eq++] = (v(x,y) - vm(x,y)).getValue();
        }

        // Manufactured solution
        for(x = 0; x < Nx; x++)
        {
            for(y = 0; y < Ny; y++)
            {
                residuals[eq++] = (u_man(x,y) - um(x,y)).getValue();
            }
        }
        for(x = 0; x < Nx; x++)
        {
            for(y = 0; y < Ny; y++)
            {
                residuals[eq++] = (v_man(x,y) - vm(x,y)).getValue();
            }
        }
    }

    void EvaluateJacobian(double time, double inverse_ts, double* values, double* derivs, double* jacobian)
    {
        t               = adouble(time);
        inverseTimeStep = inverse_ts;

        u_data      = &values[u_start_index];
        v_data      = &values[v_start_index];
        uman_data   = &values[uman_start_index];
        vman_data   = &values[vman_start_index];

        du_dt_data    = &derivs[u_start_index];
        dv_dt_data    = &derivs[v_start_index];
        duman_dt_data = &derivs[uman_start_index];
        dvman_dt_data = &derivs[vman_start_index];

        int eq = 0;
        int jac = 0;
        int x, y;

        // u component
        for(x = 1; x < Nx-1; x++)
        {
            for(y = 1; y < Ny-1; y++)
            {
                std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
                for(int ji = 0; ji < blockIndexes.size(); ji++)
                {
                    jacobianIndex = blockIndexes[ji];
                    jacobian[jac++] = (du_dt(x,y) + (duu_dx(x,y) + duv_dy(x,y)) - ni * (d2u_dx2(x,y) + d2u_dy2(x,y)) - Su(x,y)).getDerivative();
                }
            }
        }
        // u BCs
        for(x = 1; x < Nx-1; x++)
        {
            y = 0;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (u(x,y) - um(x,y)).getDerivative();
            }
        }
        for(x = 1; x < Nx-1; x++)
        {
            y = Ny-1;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (u(x,y) - um(x,y)).getDerivative();
            }
        }
        for(y = 0; y < Ny; y++)
        {
            x = 0;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (u(x,y) - um(x,y)).getDerivative();
            }
        }
        for(y = 0; y < Ny; y++)
        {
            x = Nx-1;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (u(x,y) - um(x,y)).getDerivative();
            }
        }

        // v component
        for(x = 1; x < Nx-1; x++)
        {
            for(y = 1; y < Ny-1; y++)
            {
                std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
                for(int ji = 0; ji < blockIndexes.size(); ji++)
                {
                    jacobianIndex = blockIndexes[ji];
                    jacobian[jac++] = (dv_dt(x,y) + (dvu_dx(x,y) + dvv_dy(x,y)) - ni * (d2v_dx2(x,y) + d2v_dy2(x,y)) - Sv(x,y)).getDerivative();
                }
            }
        }
        // v BCs
        for(x = 1; x < Nx-1; x++)
        {
            y = 0;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (v(x,y) - vm(x,y)).getDerivative();
            }
        }
        for(x = 1; x < Nx-1; x++)
        {
            y = Ny-1;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (v(x,y) - vm(x,y)).getDerivative();
            }
        }
        for(y = 0; y < Ny; y++)
        {
            x = 0;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (v(x,y) - vm(x,y)).getDerivative();
            }
        }
        for(y = 0; y < Ny; y++)
        {
            x = Nx-1;

            std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
            for(int ji = 0; ji < blockIndexes.size(); ji++)
            {
                jacobianIndex = blockIndexes[ji];
                jacobian[jac++] = (v(x,y) - vm(x,y)).getDerivative();
            }
        }

        // Manufactured solution
        for(x = 0; x < Nx; x++)
        {
            for(y = 0; y < Ny; y++)
            {
                std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
                for(int ji = 0; ji < blockIndexes.size(); ji++)
                {
                    jacobianIndex = blockIndexes[ji];
                    jacobian[jac++] = (u_man(x,y) - um(x,y)).getDerivative();
                }
            }
        }
        for(x = 0; x < Nx; x++)
        {
            for(y = 0; y < Ny; y++)
            {
                std::vector<int>& blockIndexes = jacobianMatrixIndexes[eq++];
                for(int ji = 0; ji < blockIndexes.size(); ji++)
                {
                    jacobianIndex = blockIndexes[ji];
                    jacobian[jac++] = (v_man(x,y) - vm(x,y)).getDerivative();
                }
            }
        }
    }
};

#endif
