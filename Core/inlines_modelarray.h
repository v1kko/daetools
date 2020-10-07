/******************************************************************
    daeModelArrayBase
*******************************************************************/
template<typename TYPE, size_t M>
class daeModelArrayBase : public daeModelArray
{
public:
    daeDeclareDynamicClass(daeModelArrayBase)
    daeModelArrayBase() : daeModelArray(M)
    {
    }
    virtual ~daeModelArrayBase(void)
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

    virtual void UpdateEquations()
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->UpdateEquations();
    }

    virtual void BuildExpressions(daeBlock* pBlock)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->BuildExpressions(pBlock);
    }

    virtual bool CheckDiscontinuities(void)
    {
        bool bReturn = false;
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
        {
            if(it->CheckDiscontinuities())
                bReturn = true;
        }
        return bReturn;
    }

    virtual void AddExpressionsToBlock(daeBlock* pBlock)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->AddExpressionsToBlock(pBlock);
    }

    virtual void ExecuteOnConditionActions(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->ExecuteOnConditionActions();
    }

    virtual void CreateOverallIndex_BlockIndex_VariableNameMap(std::map<size_t, std::pair<size_t, string> >& mapOverallIndex_BlockIndex_VariableName,
                                                               const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CreateOverallIndex_BlockIndex_VariableNameMap(mapOverallIndex_BlockIndex_VariableName, mapOverallIndex_BlockIndex);
    }

    virtual void InitializeModels(const std::string& jsonInit)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeModel(jsonInit);
    }

    virtual void CollectAllDomains(std::map<std::string, daeDomain_t*>& mapDomains) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllDomains(mapDomains);
    }

    virtual void CollectAllParameters(std::map<std::string, daeParameter_t*>& mapParameters) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllParameters(mapParameters);
    }

    virtual void CollectAllVariables(std::map<std::string, daeVariable_t*>& mapVariables) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllVariables(mapVariables);
    }

    virtual void CollectAllSTNs(std::map<std::string, daeSTN_t*>& mapSTNs) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllSTNs(mapSTNs);
    }

    virtual void CollectAllPorts(std::map<std::string, daePort_t*>& mapPorts) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllPorts(mapPorts);
    }

protected:
    virtual size_t GetTotalNumberOfVariables(void) const
    {
        size_t nNoVars = 0;
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            nNoVars += it->GetTotalNumberOfVariables();
        return nNoVars;
    }

    virtual size_t GetTotalNumberOfEquations(void) const
    {
        size_t nNoEqns = 0;
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            nNoEqns += it->GetTotalNumberOfEquations();
        return nNoEqns;
    }

    virtual void DeclareData(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->DeclareData();
    }

    virtual void DeclareEquations(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->DeclareEquations();
    }

    virtual void CreatePortConnectionEquations(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CreatePortConnectionEquations();
    }

    virtual void InitializeParameters(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeParameters();
    }

    virtual void InitializeBlockIndexes(const std::map<size_t, size_t>& mapOverallIndex_BlockIndex)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeBlockIndexes(mapOverallIndex_BlockIndex);
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

    virtual void InitializeSTNs(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeSTNs();
    }

    void InitializeExternalFunctions(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeExternalFunctions();
    }

    virtual void InitializeDEDIs(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeDEDIs();
    }

    virtual void InitializeEquations(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->InitializeEquations();
    }

    virtual void PropagateDataProxy(std::shared_ptr<daeDataProxy_t> pDataProxy)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->PropagateDataProxy(pDataProxy);
    }

    virtual void PropagateGlobalExecutionContext(daeExecutionContext* pExecutionContext)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->PropagateGlobalExecutionContext(pExecutionContext);
    }

    virtual void CollectAllSTNsAsVector(std::vector<daeSTN*>& ptrarrSTNs) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectAllSTNsAsVector(ptrarrSTNs);
    }

    virtual void CollectEquationExecutionInfosFromModels(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectEquationExecutionInfosFromModels(ptrarrEquationExecutionInfo);
    }

    virtual void CollectEquationExecutionInfosFromSTNs(std::vector<daeEquationExecutionInfo*>& ptrarrEquationExecutionInfo) const
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->CollectEquationExecutionInfosFromSTNs(ptrarrEquationExecutionInfo);
    }

    virtual void SetDefaultAbsoluteTolerances(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->SetDefaultAbsoluteTolerances();
    }

    virtual void SetDefaultInitialGuessesAndConstraints(void)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->SetDefaultInitialGuessesAndConstraints();
    }

    virtual void PropagateDomain(daeDomain& propagatedDomain)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->PropagateDomain(propagatedDomain);
    }

    virtual void PropagateParameter(daeParameter& propagatedParameter)
    {
        for(iterator it = this->m_ptrarrObjects.begin(); it != this->m_ptrarrObjects.end(); it++)
            it->PropagateParameter(propagatedParameter);
    }

protected:
    boost::multi_array<TYPE, M> m_ptrarrObjects;
};

/******************************************************************
    daeModelArray1
*******************************************************************/
template<typename TYPE>
class daeModelArray1 : public daeModelArrayBase<TYPE, 1>
{
    daeDeclareDynamicClass(daeModelArray1)

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

    virtual daeModel* GetModel(size_t n1)
    {
        return &operator()(n1);
    }

    virtual void Open(io::xmlTag_t* pTag)
    {
        daeModelArray::Open(pTag);
    }

    virtual void Save(io::xmlTag_t* pTag) const
    {
        //size_t n1;
        //string strParentName;

        daeModelArray::Save(pTag);

        //n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();

        //strParentName = "Models";
        //for(size_t i = 0; i < n1; i++)
        //{
        //	TYPE& object = this->m_ptrarrObjects[i];
        //	io::daeSaveObject<daeModel>(pTag, this->m_ptrarrDomains, &object);
        //}
    }

    virtual bool CheckObject(std::vector<string>& strarrErrors) const
    {
        string strError;

        bool bCheck = true;

    // Check base class
        if(!daeModelArray::CheckObject(strarrErrors))
            bCheck = false;

    // Check object array
        if(this->m_ptrarrObjects.num_dimensions() != this->N)
        {
            strError = "Invalid number of dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

    // Check each object
        size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
        const size_t* dimensions = this->m_ptrarrObjects.shape();

        if(n1 != dimensions[0])
        {
            strError = "Invalid dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

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

        daeModelArray::Create();

        n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();

        this->m_ptrarrObjects.resize(boost::extents[n1]);

        for(size_t i = 0; i < n1; i++)
        {
            TYPE& object = this->m_ptrarrObjects[i];
            strName = this->m_strShortName + "(" + toString<size_t>(i) + ")";
            daeDeclareAndThrowException(exNotImplemented);
            //m_pModel->AddModel(object, strName);
            object.DeclareData();
        }
    }
};

/******************************************************************
    daeModelArray2
*******************************************************************/
template<typename TYPE>
class daeModelArray2 : public daeModelArrayBase<TYPE, 2>
{
    daeDeclareDynamicClass(daeModelArray2)

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

    virtual daeModel* GetModel(size_t n1, size_t n2)
    {
        return &operator()(n1, n2);
    }

    virtual void Open(io::xmlTag_t* pTag)
    {
        daeModelArray::Open(pTag);
    }

    virtual void Save(io::xmlTag_t* pTag) const
    {
        daeModelArray::Save(pTag);
    }

    virtual bool CheckObject(std::vector<string>& strarrErrors) const
    {
        string strError;

        bool bCheck = true;

    // Check base class
        if(!daeModelArray::CheckObject(strarrErrors))
            bCheck = false;

    // Check object array
        if(this->m_ptrarrObjects.num_dimensions() != this->N)
        {
            strError = "Invalid number of dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

    // Check each object
        size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
        size_t n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
        const size_t* dimensions = this->m_ptrarrObjects.shape();

        if(n1 != dimensions[0] ||
           n2 != dimensions[1] )
        {
            strError = "Invalid dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

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

        daeModelArray::Create();

        n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
        n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();

        this->m_ptrarrObjects.resize(boost::extents[n1][n2]);

        for(size_t i = 0; i < n1; i++)
        for(size_t j = 0; j < n2; j++)
        {
            TYPE& object = this->m_ptrarrObjects[i][j];
            strName = this->m_strShortName + "(" + toString<size_t>(i) + ", " + toString<size_t>(j) + ")";
            daeDeclareAndThrowException(exNotImplemented);
            //m_pModel->AddModel(object, strName);
            object.DeclareData();
        }
    }
};

/******************************************************************
    daeModelArray3
*******************************************************************/
template<typename TYPE>
class daeModelArray3 : public daeModelArrayBase<TYPE, 3>
{
    daeDeclareDynamicClass(daeModelArray3)

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

    virtual daeModel* GetModel(size_t n1, size_t n2, size_t n3)
    {
        return &operator()(n1, n2, n3);
    }

    virtual void Open(io::xmlTag_t* pTag)
    {
        daeModelArray::Open(pTag);
    }

    virtual void Save(io::xmlTag_t* pTag) const
    {
        daeModelArray::Save(pTag);
    }

    virtual bool CheckObject(std::vector<string>& strarrErrors) const
    {
        string strError;

        bool bCheck = true;

    // Check base class
        if(!daeModelArray::CheckObject(strarrErrors))
            bCheck = false;

    // Check object array
        if(this->m_ptrarrObjects.num_dimensions() != this->N)
        {
            strError = "Invalid number of dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

    // Check each object
        size_t n1 = this->m_ptrarrDomains[0]->GetNumberOfPoints();
        size_t n2 = this->m_ptrarrDomains[1]->GetNumberOfPoints();
        size_t n3 = this->m_ptrarrDomains[2]->GetNumberOfPoints();
        const size_t* dimensions = this->m_ptrarrObjects.shape();

        if(n1 != dimensions[0] ||
           n2 != dimensions[1] ||
           n3 != dimensions[2])
        {
            strError = "Invalid dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

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

        daeModelArray::Create();

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
            //m_pModel->AddModel(object, strName);
            object.DeclareData();
        }
    }
};

/******************************************************************
    daeModelArray4
*******************************************************************/
template<typename TYPE>
class daeModelArray4 : public daeModelArrayBase<TYPE, 4>
{
    daeDeclareDynamicClass(daeModelArray4)

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

    virtual daeModel* GetModel(size_t n1, size_t n2, size_t n3, size_t n4)
    {
        return &operator()(n1, n2, n3, n4);
    }

    virtual void Open(io::xmlTag_t* pTag)
    {
        daeModelArray::Open(pTag);
    }

    virtual void Save(io::xmlTag_t* pTag) const
    {
        daeModelArray::Save(pTag);
    }

    virtual bool CheckObject(std::vector<string>& strarrErrors) const
    {
        string strError;

        bool bCheck = true;

    // Check base class
        if(!daeModelArray::CheckObject(strarrErrors))
            bCheck = false;

    // Check object array
        if(this->m_ptrarrObjects.num_dimensions() != this->N)
        {
            strError = "Invalid number of dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

    // Check each object
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
            strError = "Invalid dimensions in model array [" + this->GetCanonicalName() + "]";
            strarrErrors.push_back(strError);
            bCheck = false;
        }

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

        daeModelArray::Create();

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
            //m_pModel->AddModel(object, strName);
            object.DeclareData();
        }
    }
};

