

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 8.00.0603 */
/* at Tue Apr 18 14:20:43 2017
 */
/* Compiler settings for DAEToolsCapeOpen.idl:
    Oicf, W1, Zp8, env=Win32 (32b run), target_arch=X86 8.00.0603 
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */

#pragma warning( disable: 4049 )  /* more than 64k source lines */


/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __DAEToolsCapeOpen_i_h__
#define __DAEToolsCapeOpen_i_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __IdaeCapeThermoMaterial_FWD_DEFINED__
#define __IdaeCapeThermoMaterial_FWD_DEFINED__
typedef interface IdaeCapeThermoMaterial IdaeCapeThermoMaterial;

#endif 	/* __IdaeCapeThermoMaterial_FWD_DEFINED__ */


#ifndef __daeCapeThermoMaterial_FWD_DEFINED__
#define __daeCapeThermoMaterial_FWD_DEFINED__

#ifdef __cplusplus
typedef class daeCapeThermoMaterial daeCapeThermoMaterial;
#else
typedef struct daeCapeThermoMaterial daeCapeThermoMaterial;
#endif /* __cplusplus */

#endif 	/* __daeCapeThermoMaterial_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


#ifndef __IdaeCapeThermoMaterial_INTERFACE_DEFINED__
#define __IdaeCapeThermoMaterial_INTERFACE_DEFINED__

/* interface IdaeCapeThermoMaterial */
/* [unique][nonextensible][dual][uuid][object] */ 


EXTERN_C const IID IID_IdaeCapeThermoMaterial;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("63951F8C-5C22-46F4-B1A5-EA4E5EFFACD0")
    IdaeCapeThermoMaterial : public IDispatch
    {
    public:
    };
    
    
#else 	/* C style interface */

    typedef struct IdaeCapeThermoMaterialVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IdaeCapeThermoMaterial * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            _COM_Outptr_  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IdaeCapeThermoMaterial * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IdaeCapeThermoMaterial * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            IdaeCapeThermoMaterial * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            IdaeCapeThermoMaterial * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            IdaeCapeThermoMaterial * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [range][in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            IdaeCapeThermoMaterial * This,
            /* [annotation][in] */ 
            _In_  DISPID dispIdMember,
            /* [annotation][in] */ 
            _In_  REFIID riid,
            /* [annotation][in] */ 
            _In_  LCID lcid,
            /* [annotation][in] */ 
            _In_  WORD wFlags,
            /* [annotation][out][in] */ 
            _In_  DISPPARAMS *pDispParams,
            /* [annotation][out] */ 
            _Out_opt_  VARIANT *pVarResult,
            /* [annotation][out] */ 
            _Out_opt_  EXCEPINFO *pExcepInfo,
            /* [annotation][out] */ 
            _Out_opt_  UINT *puArgErr);
        
        END_INTERFACE
    } IdaeCapeThermoMaterialVtbl;

    interface IdaeCapeThermoMaterial
    {
        CONST_VTBL struct IdaeCapeThermoMaterialVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IdaeCapeThermoMaterial_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define IdaeCapeThermoMaterial_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define IdaeCapeThermoMaterial_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define IdaeCapeThermoMaterial_GetTypeInfoCount(This,pctinfo)	\
    ( (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo) ) 

#define IdaeCapeThermoMaterial_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    ( (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo) ) 

#define IdaeCapeThermoMaterial_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    ( (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) ) 

#define IdaeCapeThermoMaterial_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    ( (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) ) 


#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __IdaeCapeThermoMaterial_INTERFACE_DEFINED__ */



#ifndef __DAEToolsCapeOpenLib_LIBRARY_DEFINED__
#define __DAEToolsCapeOpenLib_LIBRARY_DEFINED__

/* library DAEToolsCapeOpenLib */
/* [version][uuid] */ 


EXTERN_C const IID LIBID_DAEToolsCapeOpenLib;

EXTERN_C const CLSID CLSID_daeCapeThermoMaterial;

#ifdef __cplusplus

class DECLSPEC_UUID("6DAB8251-8FC4-46AA-BA9A-3CD6B2E50696")
daeCapeThermoMaterial;
#endif
#endif /* __DAEToolsCapeOpenLib_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


