#ifndef DAE_EXAMPLE_SLAB_H
#define DAE_EXAMPLE_SLAB_H

#include "variable_types.h"

const int NP = 4;

/******************************************************************
	HeatFlux port
*******************************************************************/
class HeatFlux : public daePort
{
	daeDeclareDynamicClass(HeatFlux)
public:
	HeatFlux(std::string strName, daeePortType type, daeModel* parent) : daePort(strName, type, parent) 
	{
		AddVariable(Flux, "Flux", heat_flux);
	}

public:
	daeVariable	Flux;
};

/******************************************************************
	TPP port
*******************************************************************/
class TPP : public daePort
{
	daeDeclareDynamicClass(TPP)
public:
	TPP(std::string strName, daeePortType type, daeModel* parent) : daePort(strName, type, parent) 
	{
		AddDomain(z, "z");

		ro.DistributeOnDomain(z);
		AddVariable(ro, "ro", density);

		cp.DistributeOnDomain(z);
		AddVariable(cp, "cp", heat_capacity);

		lamb.DistributeOnDomain(z);
		AddVariable(lamb, "lamb", conductivity);

		T.DistributeOnDomain(z);
		AddVariable(T, "T", temperature);
	}

public:
	daeDomain	z;

	daeVariable ro;
	daeVariable	cp;
	daeVariable	lamb;
	daeVariable	T;
};

/******************************************************************
	Simple model
*******************************************************************/
class Simple : public daeModel
{
	daeDeclareDynamicClass(Simple)
public:
// Variables
	daeVariable ro;

public:
	Simple(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddVariable(ro, "ro", density);
	}
};


/******************************************************************
	TPP_Package model
*******************************************************************/
class TPP_Package : public daeModel
{
	daeDeclareDynamicClass(TPP_Package)
public:
// Domains
	daeDomain	z;

// Variables
	daeVariable		ro;
	daeVariable		cp;
	daeVariable		lamb;
	daeVariable		T;

// Parameters
	daeParameter	_ro;
	daeParameter	_cp;
	daeParameter	_lamb;

// Ports
	TPP			tpp;

public:
	TPP_Package(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription),
	                                                                                    tpp("tpp", eOutletPort, this)
	{
		AddParameter(_ro, "_ro", eReal);
		AddParameter(_cp, "_cp", eReal);
		AddParameter(_lamb, "_lamb", eReal);

		AddDomain(z, "z");

		ro.DistributeOnDomain(z);
		AddVariable(ro, "ro", density);

		cp.DistributeOnDomain(z);
		AddVariable(cp, "cp", heat_capacity);

		lamb.DistributeOnDomain(z);
		AddVariable(lamb, "lamb", conductivity);

		T.DistributeOnDomain(z);
		AddVariable(T, "T", temperature);

		//AddPort(tpp, "tpp", eOutletPort);
	}

	void DeclareEquations(void)
	{
		daeDEDI* nz;
		daeEquation* pFn;

		pFn = CreateEquation("Temperature");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(tpp.T(nz) - T(nz));

		pFn = CreateEquation("Ro");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(tpp.ro(nz) - ro(nz));

		pFn = CreateEquation("Cp");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(tpp.cp(nz) - cp(nz));

		pFn = CreateEquation("Lambda");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(tpp.lamb(nz) - lamb(nz));

		pFn = CreateEquation("Ro_f_T");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(ro(nz) - _ro());

		pFn = CreateEquation("Cp_f_T");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(cp(nz) - _cp());

		pFn = CreateEquation("Lambda_f_T");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(lamb(nz) - _lamb());
	}
};

/******************************************************************
	Slab model
*******************************************************************/
class Slab : public daeModel
{
public:
	daeDeclareDynamicClass(Slab)
// Domains
	daeDomain	x;
	daeDomain	y;
	daeDomain	z;

// Assigned variables
	daeVariable	fixed;

// State variables
	daeVariable	T;
	daeVariable	ro;
	daeVariable	cp;
	daeVariable	lamb;
	daeVariable ifvar;
	daeVariable ifvar2;
	daeVariable time;

	daeVariable	varN6;
	daeVariable	varTwoDomains;

	daeVariable SUM;

// Ports
//	HeatFlux	Qin;
//	HeatFlux	Qout;
	TPP			tpp;

// Models
	TPP_Package phys_prop;
//	daeDeclareModelArray1(Simple, arrayS)
//	daeDeclarePortArray3(HeatFlux, arrayHF);

public:
	Slab(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription),
													                             tpp("tpp", eInletPort, this),
													                             phys_prop("phys_prop", this)
	{
		//AddModel(phys_prop, "phys_prop");

		AddDomain(x, "x");
		AddDomain(y, "y");
		AddDomain(z, "z");

		SUM.DistributeOnDomain(z);
		AddVariable(SUM,    "SUM",    no_type);

		AddVariable(ifvar,  "ifvar1", no_type);
		AddVariable(ifvar2, "ifvar2", no_type);
		AddVariable(fixed,  "fixed",  no_type);
		AddVariable(time,   "time",   no_type);

		varTwoDomains.DistributeOnDomain(z);
		varTwoDomains.DistributeOnDomain(z);
		AddVariable(varTwoDomains, "V", density);

		ro.DistributeOnDomain(z);
		AddVariable(ro, "ro", density);

		cp.DistributeOnDomain(z);
		AddVariable(cp, "cp", heat_capacity);

		lamb.DistributeOnDomain(z);
		AddVariable(lamb, "lamb", conductivity);

		T.DistributeOnDomain(z);
		AddVariable(T, "T", temperature);

		//AddPort(tpp, "tpp", eInletPort);
	}

	void DeclareEquations(void)
	{
		daeDEDI *nz, *nz1, *nz2;
		daeEquation* pFn;

		pFn = CreateEquation("varTwoDomains");
		nz1 = pFn->DistributeOnDomain(z, eClosedClosed);
		nz2 = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( pow(varTwoDomains(nz1, nz2), 2) + 2*varTwoDomains(nz1, nz2) - 2 );

		daeIndexRange r(&z);
		pFn = CreateEquation("SUM");
		nz1 = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual(SUM(nz1) - sum(varTwoDomains.array(nz1, r)));

		pFn = CreateEquation("Fixed");
		pFn->SetResidual(0 - (-fixed()) + 1.0);

		pFn = CreateEquation("time");
		pFn->SetResidual(time.dt() - 1);

		pFn = CreateEquation("ifvar2");
		pFn->SetResidual(ifvar2() - 0);
        
		STN("TestSTN");
			STATE("First");
				pFn = CreateEquation("t_10");
				pFn->SetResidual( ifvar() - 1 );
				SWITCH_TO("Second", time() >= 10.5);
	
			STATE("Second");
				pFn = CreateEquation("t_20");
				pFn->SetResidual( ifvar() - 2 );
				SWITCH_TO("Third", time() >= 20);
	
			STATE("Third");
				pFn = CreateEquation("t_30");
				pFn->SetResidual( ifvar() - 3 );
				SWITCH_TO("Final", time() >= 30);
	
			STATE("Final");
				pFn = CreateEquation("t_40");
				pFn->SetResidual( ifvar() - 4 );
        END_STN();
		
//		IF(time() < 10);
//			pFn = CreateEquation("ifvar");
//			pFn->SetResidual(ifvar() - 0);
//			pFn = CreateEquation("ifvar2");
//			pFn->SetResidual(ifvar2() - 0);
//
//		ELSE_IF(time() >= 10 && time() < 20);
//			pFn = CreateEquation("ifvar");
//			pFn->SetResidual(ifvar() - 1);
//			pFn = CreateEquation("ifvar2");
//			pFn->SetResidual(ifvar2() - 1);
//
//		ELSE_IF(time() >= 20 && time() < 30);
//			pFn = CreateEquation("ifvar");
//			pFn->SetResidual(ifvar() - 2);
//			pFn = CreateEquation("ifvar2");
//			pFn->SetResidual(ifvar2() - 2);
//
//      ELSE_IF(time() >= 30 && time() < 40 || time() > 100 || time() > 200);
//            pFn = CreateEquation("ifvar");
//            pFn->SetResidual(ifvar() - 3);
//            pFn = CreateEquation("ifvar2");
//            pFn->SetResidual(ifvar2() - 3);
//
//		ELSE();
//			pFn = CreateEquation("ifvar");
//			pFn->SetResidual(ifvar() - 4);
//			pFn = CreateEquation("ifvar2");
//			pFn->SetResidual(ifvar2() - 4);
//
//		END_IF();

		pFn = CreateEquation("Temperature1");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) + 1e6 );

		//IF(T(0) > 290 && T(0) <= 300)
		//	pFn = AddStateEquation("Temperature1");
		//	nz = pFn->DistributeOnDomain(z, eClosedClosed);
		//	pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) + 1e6 );

		//ELSE_IF(T(0) > 280 && T(0) <= 290)
		//	pFn = AddStateEquation("Temperature2");
		//	nz = pFn->DistributeOnDomain(z, eClosedClosed);
		//	pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) + 1e6 );

		//ELSE_IF(T(0) > 270 && T(0) <= 280)
		//	pFn = AddStateEquation("Temperature3");
		//	nz = pFn->DistributeOnDomain(z, eClosedClosed);
		//	pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) + 1e6 );

		//ELSE_IF(T(0) > 260 && T(0) <= 270)
		//	pFn = AddStateEquation("Temperature4");
		//	nz = pFn->DistributeOnDomain(z, eClosedClosed);
		//	pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) + 1e6 );

		//ELSE
		//	pFn = AddStateEquation("Temperature5");
		//	nz = pFn->DistributeOnDomain(z, eClosedClosed);
		//	pFn->SetResidual( ro(nz) * cp(nz) * T.dt(nz) - lamb(nz) * T.d2(z, nz) );

		//END_IF

		pFn = CreateEquation("Temp");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( tpp.T(nz) - T(nz) );

		pFn = CreateEquation("Ro");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( tpp.ro(nz) - ro(nz) );

		pFn = CreateEquation("Cp");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( tpp.cp(nz) - cp(nz) );

		pFn = CreateEquation("Lambda");
		nz = pFn->DistributeOnDomain(z, eClosedClosed);
		pFn->SetResidual( tpp.lamb(nz) - lamb(nz) );

		ConnectPorts(&phys_prop.tpp, &tpp);
	}
};

/******************************************************************
	SlabDerived model
*******************************************************************/
class SlabDerived : public Slab
{
	daeDeclareDynamicClass(SlabDerived)
public:
	daeParameter	SlabDerived_ro;

public:
	SlabDerived(string strName, daeModel* pParent = NULL) : Slab(strName, pParent)
	{
		AddParameter(SlabDerived_ro, "SlabDerived_ro", eReal);
	}

	void DeclareEquations(void)
	{
	}

	virtual adouble	Fixed(void)
	{
        adouble ad = fixed() + 2.0;
		return ad;
	}
};

/******************************************************************
	Simulation SimulateSlab
*******************************************************************/
class SimulateSlab : public daeDynamicSimulation
{
public:
	SimulateSlab(void) : s("Slab")
	{
		SetModel(&s);
	}

public:
	void SetUpParametersAndDomains(void)
	{
		s.x.CreateDistributed(eCFDM, 2, NP, 0, 1);
		s.y.CreateDistributed(eCFDM, 2, NP, 0, 1);
		s.z.CreateDistributed(eCFDM, 2, NP, 0, 1);

		s.tpp.z.CreateDistributed(eCFDM, 2, NP, 0, 1);

		s.phys_prop.z.CreateDistributed(eCFDM, 2, NP, 0, 1);
		s.phys_prop.tpp.z.CreateDistributed(eCFDM, 2, NP, 0, 1);

		s.phys_prop._ro.SetValue(1000);
		s.phys_prop._cp.SetValue(4186);
		s.phys_prop._lamb.SetValue(1);

		s.SlabDerived_ro.SetValue(-1);
	}

	void SetUpVariables(void)
	{
		for(size_t i = 0; i < s.z.GetNumberOfPoints(); i++)
			s.T.SetInitialCondition(i, 300);

		s.time.SetInitialCondition(0);
	}

protected:
	SlabDerived s;
};


#endif
