

/******************************************************************
	daePortArrayBase
*******************************************************************/
template<typename TYPE, const int M>
class daePortArrayBase : public daePortArray
{
public:
	daeDeclareDynamicClass(daePortArrayBase)
	daePortArrayBase() : daePortArray(M)
	{
	}
	virtual ~daePortArrayBase(void)
	{
	}
	
	typedef typename boost::multi_array<TYPE, M>::size_type size_type;
	typedef typename boost::multi_array<TYPE, M>::iterator  iterator;

public:
	virtual void SetReportingOn(bool bOn)
	{
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->ReportAllVariables(bOn);
	}
	
	virtual void DetectVariableTypesForExport(std::vector<const daeVariableType*>& ptrarrVariableTypes) const
	{
		iterator it = this->m_ptrarrObjects.begin();
		if(it != this->m_ptrarrObjects.end())
			it->DetectVariableTypesForExport(ptrarrVariableTypes);
	}
	
	virtual void CleanUpSetupData(void)
	{
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->CleanUpSetupData();
	}

    virtual void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                               const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }

    virtual void CollectAllDomains(std::map<dae::string, daeDomain_t*>& mapDomains) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllDomains(mapDomains);
    }

    virtual void CollectAllParameters(std::map<dae::string, daeParameter_t*>& mapParameters) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllParameters(mapParameters);
    }

	virtual void CollectAllVariables(std::map<dae::string, daeVariable_t*>& mapVariables) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllVariables(mapVariables);
    }

protected:
	virtual void InitializeParameters(void)
	{
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->InitializeParameters();
	}
	
	virtual void InitializeVariables(void)
	{
		_currentVariablesIndex = m_nVariablesStartingIndex;
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
		{
			it->SetVariablesStartingIndex(_currentVariablesIndex);
			it->InitializeVariables();
			_currentVariablesIndex = it->_currentVariablesIndex;
		}
	}
	
	virtual void SetDefaultAbsoluteTolerances(void)
	{
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->SetDefaultAbsoluteTolerances();
	}
	
	virtual void SetDefaultInitialGuesses(void)
	{
		for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
			it->SetDefaultInitialGuesses();
	}
		
protected:
	boost::multi_array<TYPE, M> m_ptrarrObjects;
};

/******************************************************************
	daePortArray1
*******************************************************************/
template<typename TYPE>
class daePortArray1 : public daePortArrayBase<TYPE, 1>
{
	daeDeclareDynamicClass(daePortArray1)
	
public:	
	TYPE& operator()(size_t n1)
	{
		if(this->N != 1)
			daeDeclareAndThrowException(exInvalidCall);

		daeDomain* pDomain1 = this->m_ptrarrDomains[0];
		if(!pDomain1)
		{	
			daeDeclareException(exInvalidCall);
			e << "Invalid domain in model array [" << this->GetCanonicalName() << "]";
			throw e;
		}

		if(n1 < 0 || n1 >= pDomain1->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 1 [" << n1 << "] out of range (0, " << pDomain1->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}

		return this->m_ptrarrObjects[n1];
	}

	virtual daePort* GetPort(size_t n1)
	{
		return &operator()(n1);
	}

	virtual void Open(io::xmlTag_t* pTag)
	{
		this->Open(pTag);
	}

	virtual void Save(io::xmlTag_t* pTag) const
	{
		this->Save(pTag);
	}

	virtual bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!this->CheckObject(strarrErrors))
			bCheck = false;
	
	// Check object array
		if(this->m_ptrarrObjects.num_dimensions() != this->N)
		{
			strError = "Invalid number of dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check dimensions
		size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		const size_t* dimensions = this->m_ptrarrObjects.shape();

		if(n1 != dimensions[0])
		{
			strError = "Invalid dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
			
	// Check each object
		for(size_t i = 0; i < n1; i++)
		{
			TYPE& object = this->m_ptrarrObjects[i];
			if(!object.CheckObject())
				bCheck = false;
		}
		
		return bCheck;
	}
	
protected:
	virtual void Create()
	{
		size_t n1;
		string strName;

		this->Create();

		n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();

		this->m_ptrarrObjects.resize(boost::extents[n1]);

		for(size_t i = 0; i < n1; i++)
		{
			TYPE& object = this->m_ptrarrObjects[i];
			strName = this->m_strShortName + "(" + toString<size_t>(i) + ")";
			daeDeclareAndThrowException(exNotImplemented);
			//m_pModel->AddPort(object, strName, m_ePortType);
			object.DeclareData();
		}
	}
};

/******************************************************************
	daePortArray2
*******************************************************************/
template<typename TYPE>
class daePortArray2 : public daePortArrayBase<TYPE, 2>
{
	daeDeclareDynamicClass(daePortArray2)
	
public:	
	TYPE& operator()(size_t n1, size_t n2)
	{
		if(this->N != 2)
			daeDeclareAndThrowException(exInvalidCall);

		daeDomain* pDomain1 = this->m_ptrarrDomains[0];
		daeDomain* pDomain2 = this->m_ptrarrDomains[1];
		if(!pDomain1 || !pDomain2)
		{	
			daeDeclareException(exInvalidCall);
			e << "Invalid domain in model array [" << this->GetCanonicalName() << "]";
			throw e;
		}

		if(n1 < 0 || n1 >= pDomain1->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 1 [" << n1 << "] out of range (0, " << pDomain1->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n2 < 0 || n2 >= pDomain2->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 2 [" << n2 << "] out of range (0, " << pDomain2->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}

		return this->m_ptrarrObjects[n1][n2];
	}

	virtual daePort* GetPort(size_t n1, size_t n2)
	{
		return &operator()(n1, n2);
	}

	virtual void Open(io::xmlTag_t* pTag)
	{
		this->Open(pTag);
	}

	virtual void Save(io::xmlTag_t* pTag) const
	{
		this->Save(pTag);
	}

	virtual bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!this->CheckObject(strarrErrors))
			bCheck = false;
	
	// Check object array
		if(this->m_ptrarrObjects.num_dimensions() != this->N)
		{
			strError = "Invalid number of dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check dimensions
		size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		size_t n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
		const size_t* dimensions = this->m_ptrarrObjects.shape();
		
		if(n1 != dimensions[0] ||
		   n2 != dimensions[1] )
		{
			strError = "Invalid dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
			
	// Check each object
		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j];
			if(!object.CheckObject())
				bCheck = false;
		}
		
		return bCheck;
	}

protected:
	virtual void Create()
	{
		size_t n1, n2;
		string strName;

		this->Create();

		n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();

		this->m_ptrarrObjects.resize(boost::extents[n1][n2]);

		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j];
			strName = this->m_strShortName + "(" + toString<size_t>(i) + ", " + toString<size_t>(j) + ")";
			daeDeclareAndThrowException(exNotImplemented);
			//m_pModel->AddPort(object, strName, m_ePortType);
			object.DeclareData();
		}
	}
};

/******************************************************************
	daePortArray3
*******************************************************************/
template<typename TYPE>
class daePortArray3 : public daePortArrayBase<TYPE, 3>
{
	daeDeclareDynamicClass(daePortArray3)
		
public:	
	TYPE& operator()(size_t n1, size_t n2, size_t n3)
	{
		if(this->N != 3)
			daeDeclareAndThrowException(exInvalidCall);

		daeDomain* pDomain1 = this->m_ptrarrDomains[0];
		daeDomain* pDomain2 = this->m_ptrarrDomains[1];
		daeDomain* pDomain3 = this->m_ptrarrDomains[2];
		if(!pDomain1 || !pDomain2 || !pDomain3)
		{	
			daeDeclareException(exInvalidCall);
			e << "Invalid domain in model array [" << this->GetCanonicalName() << "]";
			throw e;
		}

		if(n1 < 0 || n1 >= pDomain1->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 1 [" << n1 << "] out of range (0, " << pDomain1->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n2 < 0 || n2 >= pDomain2->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 2 [" << n2 << "] out of range (0, " << pDomain2->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n3 < 0 || n3 >= pDomain3->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 3 [" << n3 << "] out of range (0, " << pDomain3->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}

		return this->m_ptrarrObjects[n1][n2][n3];
	}

	virtual daePort* GetPort(size_t n1, size_t n2, size_t n3)
	{
		return &operator()(n1, n2, n3);
	}

	virtual void Open(io::xmlTag_t* pTag)
	{
		this->Open(pTag);
	}

	virtual void Save(io::xmlTag_t* pTag) const
	{
		this->Save(pTag);
	}

	virtual bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!this->CheckObject(strarrErrors))
			bCheck = false;
	
	// Check object array
		if(this->m_ptrarrObjects.num_dimensions() != this->N)
		{
			strError = "Invalid number of dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check dimensions
		size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		size_t n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
		size_t n3 = this->m_ptrarrDomains[2]->GetNumberOfPoints();
		const size_t* dimensions = this->m_ptrarrObjects.shape();
		
		if(n1 != dimensions[0] ||
		   n2 != dimensions[1] ||
		   n3 != dimensions[2] )
		{
			strError = "Invalid dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
			
	// Check each object
		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		for(size_t k = 0; k < n3; k++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j][k];
			if(!object.CheckObject())
				bCheck = false;
		}
		
		return bCheck;
	}

protected:
	virtual void Create()
	{
		size_t n1, n2, n3;
		string strName;

		this->Create();

		n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
		n3 = this->m_ptrarrDomains[2]->GetNumberOfPoints();

		this->m_ptrarrObjects.resize(boost::extents[n1][n2][n3]);

		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		for(size_t k = 0; k < n3; k++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j][k];
			strName = this->m_strShortName + "(" + toString<size_t>(i) + ", " + toString<size_t>(j) + ", " + toString<size_t>(k) + ")";
			daeDeclareAndThrowException(exNotImplemented);
			//m_pModel->AddPort(object, strName, m_ePortType);
			object.DeclareData();
		}
	}
};

/******************************************************************
	daePortArray4
*******************************************************************/
template<typename TYPE>
class daePortArray4 : public daePortArrayBase<TYPE, 4>
{
	daeDeclareDynamicClass(daePortArray4)
		
public:
	TYPE& operator()(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		if(this->N != 4)
			daeDeclareAndThrowException(exInvalidCall);

		daeDomain* pDomain1 = this->m_ptrarrDomains[0];
		daeDomain* pDomain2 = this->m_ptrarrDomains[1];
		daeDomain* pDomain3 = this->m_ptrarrDomains[2];
		daeDomain* pDomain4 = this->m_ptrarrDomains[3];
		if(!pDomain1 || !pDomain2 || !pDomain3 || !pDomain4)
		{	
			daeDeclareException(exInvalidCall);
			e << "Invalid domain in model array [" << this->GetCanonicalName() << "]";
			throw e;
		}

		if(n1 < 0 || n1 >= pDomain1->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 1 [" << n1 << "] out of range (0, " << pDomain1->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n2 < 0 || n2 >= pDomain2->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 2 [" << n2 << "] out of range (0, " << pDomain2->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n3 < 0 || n3 >= pDomain3->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 3 [" << n3 << "] out of range (0, " << pDomain3->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}
		if(n4 < 0 || n4 >= pDomain4->GetNumberOfPoints())
		{	
			daeDeclareException(exOutOfBounds);
			e << "Index 4 [" << n4 << "] out of range (0, " << pDomain4->GetNumberOfPoints()-1 << ") in [" << this->GetCanonicalName() << "]";
			throw e;
		}

		return this->m_ptrarrObjects[n1][n2][n3][n4];
	}

	virtual daePort* GetPort(size_t n1, size_t n2, size_t n3, size_t n4)
	{
		return &operator()(n1, n2, n3, n4);
	}

	virtual void Open(io::xmlTag_t* pTag)
	{
		this->Open(pTag);
	}

	virtual void Save(io::xmlTag_t* pTag) const
	{
		this->Save(pTag);
	}

	virtual bool CheckObject(std::vector<string>& strarrErrors) const
	{
		string strError;
	
		bool bCheck = true;
	
	// Check base class
		if(!this->CheckObject(strarrErrors))
			bCheck = false;
	
	// Check object array
		if(this->m_ptrarrObjects.num_dimensions() != this->N)
		{
			strError = "Invalid number of dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
		
	// Check dimensions
		size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		size_t n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
		size_t n3 = this->m_ptrarrDomains[2]->GetNumberOfPoints();
		size_t n4 = this->m_ptrarrDomains[3]->GetNumberOfPoints();
		const size_t* dimensions = this->m_ptrarrObjects.shape();
		
		if(n1 != dimensions[0] ||
		   n2 != dimensions[1] ||
		   n3 != dimensions[2] ||
		   n4 != dimensions[3] )
		{
			strError = "Invalid dimensions in port array [" + this->GetCanonicalName() + "]";
			strarrErrors.push_back(strError);
			bCheck = false;
		}
			
	// Check each object
		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		for(size_t k = 0; k < n3; k++)
		for(size_t l = 0; l < n4; l++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j][k][l];
			if(!object.CheckObject())
				bCheck = false;
		}
		
		return bCheck;
	}

protected:	
	virtual void Create()
	{
		size_t n1, n2, n3, n4;
		string strName;

		this->Create();

		n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
		n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
		n3 = this->m_ptrarrDomains[2]->GetNumberOfPoints();
		n4 = this->m_ptrarrDomains[3]->GetNumberOfPoints();

		this->m_ptrarrObjects.resize(boost::extents[n1][n2][n3][n4]);

		for(size_t i = 0; i < n1; i++)
		for(size_t j = 0; j < n2; j++)
		for(size_t k = 0; k < n3; k++)
		for(size_t l = 0; l < n4; l++)
		{
			TYPE& object = this->m_ptrarrObjects[i][j][k][l];
			strName = this->m_strShortName + "(" + toString<size_t>(i) + ", " + toString<size_t>(j) + ", " + toString<size_t>(k) + ", " + toString<size_t>(l) + ")";
			daeDeclareAndThrowException(exNotImplemented);
			//m_pModel->AddPort(object, strName, m_ePortType);
			object.DeclareData();
		}
	}
};
