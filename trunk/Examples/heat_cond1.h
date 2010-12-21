#ifndef DAE_EXAMPLE_HEAT_CONDUCTION1_H
#define DAE_EXAMPLE_HEAT_CONDUCTION1_H

#include "variable_types.h"

/******************************************************************
	modTutorial3 model
*******************************************************************/
const size_t Nx = 10;

class modTutorial3 : public daeModel
{
	daeDeclareDynamicClass(modTutorial3)
public:
	daeDomain    x, y;
	daeParameter Qb, Qt, ro, cp, k;
	daeVariable	 T, Tres, Tres_arr, Tres_arr2, Ty, A, B;

public:
	modTutorial3(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription),
	// Domains:
		x("x", this, "X axis domain"),
		y("y", this, "Y axis domain"),
		
	// Parameters:
		Qb("Q_b",	   eReal, this, "Heat flux at the bottom edge of the plate, W/m2"),
		Qt("Q_t",      eReal, this, "Heat flux at the top edge of the plate, W/m2"),
		ro("&rho;",    eReal, this, "Density of the plate, kg/m3"),
		cp("c_p",      eReal, this, "Specific heat capacity of the plate, J/kgK"),
		k ("&lambda;", eReal, this, "Thermal conductivity of the plate, W/mK"),
		
	// Variables:
		T   ("T",     temperature, this, "Temperature of the plate, K"),
		
//		A  ("A",             temperature, this, "A, K"),
//		B  ("B",             temperature, this, "B, K"),
		Ty  ("Ty",           temperature, this, "Ty, K"),
		Tres("T_res",        temperature, this, "The function result"),
		Tres_arr("Tres_arr", temperature, this, "The array function result"),
		Tres_arr2("Tres_arr2", temperature, this, "The array function result")
/*
		Tplus("T_plus", temperature, this, "The "),
		Tminus("T_minus", temperature, this, "The "),
		Tmulty("T_multy", temperature, this, "The "),
		Tdivide("T_divide", temperature, this, "The "),
		
		Tave("T_ave", temperature, this, "The average"),
		Tsum("T_sum", temperature, this, "The sum"),
		Tpro("T_pro", temperature, this, "The product"),
*/
	{
		T.DistributeOnDomain(x);
		T.DistributeOnDomain(y);
		
		Ty.DistributeOnDomain(y);		
	}

	void DeclareEquations(void)
	{
		daeDEDI *nx, *ny;
		daeEquation* eq;

        eq = CreateEquation("HeatBalance", "Heat balance equation. Valid on the open x and y domains");
        nx = eq->DistributeOnDomain(x, eOpenOpen);
        ny = eq->DistributeOnDomain(y, eOpenOpen);
        eq->SetResidual( ro() * cp() * T.dt(nx, ny) - k() * (T.d2(x, nx, ny) + T.d2(y, nx, ny)) );

        eq = CreateEquation("BC_bottom", "Boundary conditions for the bottom edge");
        nx = eq->DistributeOnDomain(x, eClosedClosed);
        ny = eq->DistributeOnDomain(y, eLowerBound);
        eq->SetResidual( - k() * T.d(y, nx, ny) - Qb() );

        eq = CreateEquation("BC_top", "Boundary conditions for the top edge");
        nx = eq->DistributeOnDomain(x, eClosedClosed);
        ny = eq->DistributeOnDomain(y, eUpperBound);
        eq->SetResidual( - k() * T.d(y, nx, ny) - Qt() );

        eq = CreateEquation("BC_left", "Boundary conditions at the left edge");
        nx = eq->DistributeOnDomain(x, eLowerBound);
        ny = eq->DistributeOnDomain(y, eOpenOpen);
        eq->SetResidual( T.d(x, nx, ny) );

        eq = CreateEquation("BC_right", "Boundary conditions for the right edge");
        nx = eq->DistributeOnDomain(x, eUpperBound);
        ny = eq->DistributeOnDomain(y, eOpenOpen);
        eq->SetResidual( T.d(x, nx, ny) );

        daeIndexRange xr(&x);
        daeIndexRange yr(&y);

        eq = CreateEquation("T_ave", "The average temperature of the plate");
		eq->SetResidual( Tres() - average( T.array(xr, yr) ) - x[0] );
        
        eq = CreateEquation("Tres_array", "The array function result");
		eq->SetResidual( Tres_arr() - dt(T(1,1) / T(1,2)) );

        eq = CreateEquation("Tres_array2", "The array function result2");  
//		eq->SetResidual( Tres_arr2() - ( T(1,1)*T.dt(1,2) + T(1,2)*T.dt(1,1) ) );
		eq->SetResidual( Tres_arr2() - sum( k() * T.array(xr, 0) ) );
  
        //eq = CreateEquation("T_ave", "The average temperature of the plate");
        //eq->SetResidual( Tave() - average(T.array(xr, yr)) );

        //eq = CreateEquation("T_sum", "The sum of the plate temperatures");
        //eq->SetResidual( Tsum() + k() * sum(T.d_array(y, xr, 0)) );
	}
};

class simTutorial3 : public daeDynamicSimulation
{
public:
	modTutorial3 m;
	
public:
	simTutorial3(void) : m("Tutorial_3")
	{
		SetModel(&m);
	}

public:
	void SetUpParametersAndDomains(void)
	{
        int n = 10;
        
        m.x.CreateDistributed(eCFDM, 2, n, 0, 0.1);
        m.y.CreateDistributed(eCFDM, 2, n, 0, 0.1);
        
        m.ro.SetValue(8960);
        m.cp.SetValue(385);
        m.k.SetValue(401);

        m.Qb.SetValue(1e6);
        m.Qt.SetValue(0);
	}

	void SetUpVariables(void)
	{
		//m.A.AssignValue(8);
		//m.B.AssignValue(2);
		for(size_t iy = 0; iy < m.x.GetNumberOfPoints(); iy++)
			m.Ty.AssignValue(iy, 100 + iy);

		for(size_t ix = 1; ix < m.x.GetNumberOfPoints()-1; ix++)
			for(size_t iy = 1; iy < m.x.GetNumberOfPoints()-1; iy++)
				m.T.SetInitialCondition(ix, iy, 300);
	}
};



/******************************************************************
	modRoberts model
*******************************************************************/
const daeVariableType ty1("ty1", "-", -1.0e+100, 1.0e+100,  1.0, 1e-08);
const daeVariableType ty2("ty2", "-", -1.0e+100, 1.0e+100,  1.0, 1e-14);
const daeVariableType ty3("ty3", "-", -1.0e+100, 1.0e+100,  1.0, 1e-06);

class modRoberts : public daeModel
{
	daeDeclareDynamicClass(modRoberts)
public:
	daeVariable	p1, p2, p3, y1, y2, y3;

public:
	modRoberts(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription),
		y1("y1", ty1,     this, ""),
		y2("y2", ty2,     this, ""),
		y3("y3", ty3,     this, ""),
		p1("p1", no_type, this, ""),
		p2("p2", no_type, this, ""),
		p3("p3", no_type, this, "")
	{
	}
	
	void DeclareEquations(void)
	{
		daeEquation* eq;

        eq = CreateEquation("Equation1", "");
        eq->SetResidual( y1.dt() + p1()*y1() - p2()*y2()*y3() );

        eq = CreateEquation("Equation2", "");
        eq->SetResidual( y2.dt() - p1()*y1() + p2()*y2()*y3() + p3()*y2()*y2() );

        eq = CreateEquation("Equation3", "");
        eq->SetResidual( y1() + y2() + y3() - 1 );

		// dy1/dt = -p1*y1 + p2*y2*y3
		// dy2/dt =  p1*y1 - p2*y2*y3 - p3*y2**2
		//      0 = y1 + y2 + y3 - 1
	}
};

class simRoberts : public daeDynamicSimulation
{
public:
	modRoberts m;
	
public:
	simRoberts(void) : m("simRoberts")
	{
		SetModel(&m);
	}

public:
	void SetUpParametersAndDomains(void)
	{
	}

	void SetUpVariables(void)
	{
	//	p1=0.04, p2=1e4, p3=3e7
        m.p1.AssignValue(0.04);
        m.p2.AssignValue(1e4);
        m.p3.AssignValue(3e7);
		
	// y1 = 1, y2 = y3 = 0
		m.y1.SetInitialCondition(1);
		m.y2.SetInitialCondition(0);
	}
};



/******************************************************************
	modGradients model
*******************************************************************/
class modGradients : public daeModel
{
	daeDeclareDynamicClass(modGradients)
public:
	daeVariable	p1, p2, p3, y1, y2, y3;

public:
	modGradients(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription),
		y1("y1", ty1,     this, ""),
		y2("y2", ty2,     this, ""),
		y3("y3", ty3,     this, ""),
		p1("p1", no_type, this, ""),
		p2("p2", no_type, this, ""),
		p3("p3", no_type, this, "")
	{
	}
	
	void DeclareEquations(void)
	{
		daeEquation* eq;

        eq = CreateEquation("Equation1", "");
        eq->SetResidual( y1() - 1*p1() - 2*p2() - 3*p3() );

        eq = CreateEquation("Equation2", "");
        eq->SetResidual( y2() - 4*p1() - 5*p2() - 6*p3() );

        eq = CreateEquation("Equation3", "");
        eq->SetResidual( y3() - 7*p1() - 8*p2() - 9*p3() );
	}
};

class simGradients : public daeDynamicSimulation
{
public:
	modGradients m;
	
public:
	simGradients(void) : m("simGradients")
	{
		SetModel(&m);
	}

public:
	void SetUpParametersAndDomains(void)
	{
	}

	void SetUpVariables(void)
	{
        m.p1.AssignValue(0.1);
        m.p2.AssignValue(0.01);
        m.p3.AssignValue(0.001);
		
	}
};

#endif
