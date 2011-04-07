#include "stdafx.h"
#include "xmlfunctions.h"
#include "helpers.h"

namespace dae 
{
namespace xml
{
/*********************************************************************************************
    xmlContentCreator
*********************************************************************************************/
xmlTag_t* xmlContentCreator::Constant(xmlTag_t* parent, real_t value)
{
	daeDeclareAndThrowException(exNotImplemented)
	return NULL;
	
//	string strValue;
//	string strName   = "cn";
//	string strValue1 = toStringFormatted<real_t>(value);
//	string strValue2 = toStringFormatted<real_t>(value, 30, 14, false);
	
//	LTrim(strValue1, ' ');
//	LTrim(strValue2, ' ');

//	if(strValue1 == strValue2) //strValue1.compare(strValue2) == 0)
//		strValue = strValue1;
//	else
//		strValue = strValue2;
	
//	if(value < 0)
//		strValue = string("(") + strValue + string(")");
//	return parent->AddTag(strName, strValue);
}

xmlTag_t* xmlContentCreator::Variable(xmlTag_t* parent, 
									  string name, 
									  const vector<string>& domains)
{
	daeDeclareAndThrowException(exNotImplemented)
	return NULL;
	
//	string strName;
//	string strValue;
//	xmlTag_t* ci;
//	xmlTag_t* apply;
//	xmlTag_t* returnTag;
//
//	if(domains.size() > 0)
//	{
//		strName  = "apply";
//		strValue.empty();
//		apply = parent->AddTag(strName, strValue);
//
//		strName  = "ci";
//		strValue = name;
//		ci = apply->AddTag(strName, strValue);
//
//		strName  = "cn";
//		for(size_t i = 0; i < domains.size(); i++)
//		{
//			apply->AddTag(strName, domains[i]);
//		}
//
//		returnTag = apply;
//	}
//	else
//	{
//		strName  = "ci";
//		strValue = name;
//		ci = parent->AddTag(strName, strValue);
//
//		returnTag = ci;
//	}
//
//	strName  = "type";
//	strValue = "function";
//	ci->AddAttribute(strName, strValue);
//
//	strName  = "Index";
//	strValue = toString<size_t>(index);
//	ci->AddAttribute(strName, strValue);
//
//	return returnTag;
}

xmlTag_t* xmlContentCreator::TimeDerivative(xmlTag_t* parent, 
											size_t order, 
											string name, 
											const vector<string>& domains)
{
	daeDeclareAndThrowException(exNotImplemented)
	return NULL;
	
//	string strName;
//	string strValue;
//	xmlTag_t* apply;
//	xmlTag_t* degree;
//	xmlTag_t* bvar;
//	xmlTag_t* diff;
//
//	strName  = "apply";
//	strValue = "";
//	apply = parent->AddTag(strName, strValue);
//
//		strName  = "diff";
//		strValue = "";
//		diff = apply->AddTag(strName, strValue);
//
//		strName  = "bvar";
//		strValue = "";
//		bvar = apply->AddTag(strName, strValue);
//
//			strName  = "ci";
//			strValue = "t";
//			bvar->AddTag(strName, strValue);
//
//			if(order > 1)
//			{
//				strName  = "degree";
//				strValue = "";
//				degree = bvar->AddTag(strName, strValue);
//
//				strName  = "cn";
//				strValue = toString<size_t>(order);
//				degree->AddTag(strName, strValue);
//			}
//
//		xmlContentCreator::Variable(apply, name, domains, index);
//
//	return apply;
}

xmlTag_t* xmlContentCreator::PartialDerivative(xmlTag_t* parent,
											   size_t order, 
											   string name, 
											   string domain, 
											   const vector<string>& domains)
{
	daeDeclareAndThrowException(exNotImplemented)
	return NULL;
	
//	string strName;
//	string strValue;
//	xmlTag_t* apply;
//	xmlTag_t* bvar;
//	xmlTag_t* degree;
//	xmlTag_t* partialdiff;
//
//	strName  = "apply";
//	strValue = "";
//	apply = parent->AddTag(strName, strValue);
//		//strName  = "Kind";
//		//strValue = "PartialDerivative";
//		//bvar->AddAttribute(strName, strValue);
//
//	strName  = "partialdiff";
//	strValue = "";
//	partialdiff = apply->AddTag(strName, strValue);
//
//	strName  = "bvar";
//	strValue = "";
//	bvar = apply->AddTag(strName, strValue);
//
//	strName  = "ci";
//	strValue = domain;
//	bvar->AddTag(strName, strValue);
//
//	if(order > 1)
//	{
//		strName  = "degree";
//		strValue = "";
//		degree = bvar->AddTag(strName, strValue);
//
//		strName  = "cn";
//		strValue = toString<size_t>(order);
//		degree->AddTag(strName, strValue);
//	}
//
//	xmlContentCreator::Variable(apply, name, domains, index);
//
//	return apply;
}
	
xmlTag_t* xmlContentCreator::Domain(xmlTag_t* parent, 
									string name, 
									string strIndex)
{
	daeDeclareAndThrowException(exNotImplemented)
	return NULL;
}

//xmlTag_t* xmlContentCreator::Function(string fun, xmlTag_t* parent)
//{
//	string strName;
//	string strValue;
//	xmlTag_t* apply;
//
//	strName  = "apply";
//	strValue = "";
//	apply = parent->AddTag(strName, strValue);
//
//	strName = fun;
//	strValue = "";
//	apply->AddTag(strName, strValue);
//
//	return apply;
//}


/*********************************************************************************************
    xmlPresentationCreator
*********************************************************************************************/
xmlTag_t* xmlPresentationCreator::Constant(xmlTag_t* parent, real_t value)
{
	string strValue;
	string strName  = "mn";

	if( (real_t)((int)value) == value ) // it is an integer
	{
		strValue = toString<int>(value);
	}
	else		
	{
		if(typeid(real_t) == typeid(double))
			strValue = toStringFormatted<real_t>(value, -1, 16, false);
		else
			strValue = toStringFormatted<real_t>(value, -1, 7, false);
		LTrim(strValue, ' ');
		RTrim(strValue, '0');
	}

	if(value < 0)
		strValue = string("(") + strValue + string(")");
	
	xmlTag_t* temp = parent->AddTag(strName, strValue);
	//temp->AddAttribute(string("mathvariant"), string("italic"));
	
	return temp;
}

//struct daeDecoratedName
//{
//public:	
//	daeDecoratedName(const string& name, const string& decorated)
//	{
//		Name      = name;
//		Decorated = decorated;
//	}
//
//	std::string Name;
//	std::string Decorated;	
//};
//
//daeDecoratedName g_DecoratedNames[] = { 
//										daeDecoratedName("alpha",	"&alpha;"),
//									    daeDecoratedName("beta",	"&beta;"),
//										daeDecoratedName("gamma",	"&gamma;"),
//										daeDecoratedName("delta",	"&delta;"),
//										daeDecoratedName("epsilon", "&epsilon;"),
//										daeDecoratedName("eta",		"&eta;"),
//										daeDecoratedName("kappa",	"&kappa;"),
//										daeDecoratedName("lampbda", "&lampbda;"),
//										daeDecoratedName("mu",		"&mu;"),
//										daeDecoratedName("nu",		"&nu;"),
//										daeDecoratedName("ksi",		"&ksi;"),
//										daeDecoratedName("pi",		"&pi;"),
//										daeDecoratedName("rho",		"&rho;"),
//										daeDecoratedName("sigma",	"&sigma;"),
//										daeDecoratedName("tau",		"&tau;"),
//										daeDecoratedName("phi",		"&phi;"),
//										daeDecoratedName("chi",		"&chi;"),
//										daeDecoratedName("psi",		"&psi;"),
//										daeDecoratedName("omega",	"&omega;"),
//									    
//										daeDecoratedName("Alpha",	"&Alpha;"),
//									    daeDecoratedName("Beta",	"&Bbeta;"),
//										daeDecoratedName("Gamma",	"&Gamma;"),
//										daeDecoratedName("Delta",	"&Delta;"),
//										daeDecoratedName("Epsilon", "&Epsilon;"),
//										daeDecoratedName("Eta",		"&Eta;"),
//										daeDecoratedName("Kappa",	"&Kappa;"),
//										daeDecoratedName("Lampbda", "&Lampbda;"),
//										daeDecoratedName("Mu",		"&Mu;"),
//										daeDecoratedName("Nu",		"&Nu;"),
//										daeDecoratedName("Ksi",		"&Ksi;"),
//										daeDecoratedName("Pi",		"&Pi;"),
//										daeDecoratedName("Rho",		"&Rho;"),
//										daeDecoratedName("Sigma",	"&Sigma;"),
//										daeDecoratedName("Tau",		"&Tau;"),
//										daeDecoratedName("Phi",		"&Phi;"),
//										daeDecoratedName("Chi",		"&Chi;"),
//										daeDecoratedName("Psi",		"&Psi;"),
//										daeDecoratedName("Omega",	"&Omega;"),
//									};

void xmlPresentationCreator::WrapIdentifier(xmlTag_t* parent, string name)
{
	xmlTag_t *temp, *msub, *mrow;
	string strName, strRest;
	string::size_type iUnderscoreFound, iAmpersendFound, iDotFound, iSemicolonFound;

	iUnderscoreFound = name.find('_');
	iDotFound        = name.find('.');
	iAmpersendFound  = name.find('&');
	
	if((iUnderscoreFound == std::string::npos) && 
	   (iDotFound        == std::string::npos) &&
	   (iAmpersendFound  == std::string::npos) ) // _.& not found or the empty string
	{
		strName = name;
		temp = parent->AddTag(string("mi"), strName);
		temp->AddAttribute(string("mathvariant"), string("italic"));
	}
	else
	{
		if(iDotFound != std::string::npos) // Canonical name
		{
			vector<string> strParse;
			strParse = ParseString(name, '.');
			mrow = parent->AddTag(string("mrow"), string(""));
			for(size_t i = 0; i < strParse.size(); i++)
			{
				if(i != 0)
					mrow->AddTag(string("mo"), string("."));
					
				xmlPresentationCreator::WrapIdentifier(mrow, strParse[i]);
			}
			
		}
		else if(iUnderscoreFound != std::string::npos) // Simple name: _ found
		{
			strName = name.substr(0, iUnderscoreFound);              // H_1 -> H
			strRest = name.substr(iUnderscoreFound+1, string::npos); // H_1 -> 1
			
			msub = parent->AddTag(string("msub"), string(""));
				mrow = msub->AddTag(string("mrow"), string(""));
				xmlPresentationCreator::WrapIdentifier(mrow, strName);				
				mrow = msub->AddTag(string("mrow"), string(""));
				xmlPresentationCreator::WrapIdentifier(mrow, strRest);
		}
		else if(iAmpersendFound != std::string::npos) // Simple name: & found
		{
			iSemicolonFound = name.find(';');
			if(iAmpersendFound == 0) // & is the first character
			{
				strName = name.substr(iAmpersendFound, iSemicolonFound+1); // &Delta;H1 -> &Delta; 
				strRest = name.substr(iSemicolonFound+1, string::npos);    // &Delta;H1 -> H1
			}
			else // & is not the first character
			{
				strName = name.substr(0, iAmpersendFound);            // HH&Delta;H1 -> HH
				strRest = name.substr(iAmpersendFound, string::npos); // HH&Delta;H1 -> &Delta;H1
			}
			
			temp = parent->AddTag(string("mi"), strName);
			temp->AddAttribute(string("mathvariant"), string("italic"));
				
			xmlPresentationCreator::WrapIdentifier(parent, strRest);
		}
	}
}

xmlTag_t* xmlPresentationCreator::Variable(xmlTag_t* parent, 
										   string name, 
										   const vector<string>& domains)
{
	xmlTag_t* mrow, *temp, *msub;
	string strName, strSub;
	string::size_type iFound;

	mrow = parent->AddTag(string("mrow"), string(""));

		xmlPresentationCreator::WrapIdentifier(mrow, name);

		if(domains.size() > 0)
		{
			mrow->AddTag(string("mo"), string("("));
			for(size_t i = 0; i < domains.size(); i++)
			{
				//temp = mrow->AddTag(string("mi"), domains[i]);
				//temp->AddAttribute(string("mathvariant"), string("italic"));
				xmlPresentationCreator::WrapIdentifier(mrow, domains[i]);
				
				if(i != domains.size()-1)
					mrow->AddTag(string("mo"), string(","));
			}

			mrow->AddTag(string("mo"), string(")"));
		}

	return parent;
}

xmlTag_t* xmlPresentationCreator::Domain(xmlTag_t* parent, 
										 string name, 
										 string strIndex)
{
	xmlTag_t *mrow, *temp;
	string strName;
	string strValue;

	
	mrow = parent->AddTag(string("mrow"), string(""));
	
		//temp = mrow->AddTag(string("mi"), name);
		//temp->AddAttribute(string("mathvariant"), string("italic"));
		xmlPresentationCreator::WrapIdentifier(mrow, name);
		
		mrow->AddTag(string("mo"), string("["));
		mrow->AddTag(string("mi"), strIndex);
		mrow->AddTag(string("mo"), string("]"));

	return parent;
}

xmlTag_t* xmlPresentationCreator::TimeDerivative(xmlTag_t* parent, 
												 size_t order, 
												 string name, 
												 const vector<string>& domains)
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2;

	strName  = "mfrac";
	strValue = "";
	mfrac = parent->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

	if(order == 1)
	{
        strName  = "mo";
        strValue = "d"; // Should be &dd; but it does not show up correctly in windows
		mrow1->AddTag(strName, strValue);

		xmlPresentationCreator::Variable(mrow1, name, domains);
	}
	else
	{
		strName  = "msup";
		strValue = "";
		msup = mrow1->AddTag(strName, strValue);
			strName  = "mo";
            strValue = "d";
			msup->AddTag(strName, strValue);

			strName  = "mn";
			strValue = "2";
			msup->AddTag(strName, strValue);

		xmlPresentationCreator::Variable(mrow1, name, domains);
	}

	strName  = "mrow";
	strValue = "";
	mrow2 = mfrac->AddTag(strName, strValue);

	if(order == 1)
	{
        strName  = "mo";
        strValue = "d";
		mrow2->AddTag(strName, strValue);

		strName  = "mi";
		strValue = "t";
		mrow2->AddTag(strName, strValue);
	}
	else
	{
            strName  = "mo";
            strValue = "d";
			mrow2->AddTag(strName, strValue);

			strName  = "msup";
			strValue = "";
			msup = mrow2->AddTag(strName, strValue);

				strName  = "mi";
				strValue = "t";
				msup->AddTag(strName, strValue);

				strName  = "mn";
				strValue = "2";
				msup->AddTag(strName, strValue);
	}

	return mfrac;
}

xmlTag_t* xmlPresentationCreator::PartialDerivative(xmlTag_t* parent, 
													size_t order,
													string name, 
													string domain, 
													const vector<string>& domains)
{
	string strName, strValue;
	io::xmlTag_t *mfrac, *msup, *mrow1, *mrow2;

	strName  = "mfrac";
	strValue = "";
	mfrac = parent->AddTag(strName, strValue);

	strName  = "mrow";
	strValue = "";
	mrow1 = mfrac->AddTag(strName, strValue);

		if(order == 1)
		{
			strName  = "mo";
			strValue = "&PartialD;";
			mrow1->AddTag(strName, strValue);

			xmlPresentationCreator::Variable(mrow1, name, domains);
		}
		else
		{
			strName  = "msup";
			strValue = "";
			msup = mrow1->AddTag(strName, strValue);
				strName  = "mo";
				strValue = "&PartialD;";
				msup->AddTag(strName, strValue);

				strName  = "mn";
				strValue = "2";
				msup->AddTag(strName, strValue);
			xmlPresentationCreator::Variable(mrow1, name, domains);
		}

	strName  = "mrow";
	strValue = "";
	mrow2 = mfrac->AddTag(strName, strValue);

	if(order == 1)
	{
		strName  = "mo";
		strValue = "&PartialD;";
		mrow2->AddTag(strName, strValue);

		strName  = "mi";
		strValue = domain;
		//mrow2->AddTag(strName, strValue);
		xmlPresentationCreator::WrapIdentifier(mrow2, domain);
	}
	else
	{
			strName  = "mo";
			strValue = "&PartialD;";
			mrow2->AddTag(strName, strValue);

			strName  = "msup";
			strValue = "";
			msup = mrow2->AddTag(strName, strValue);

				strName  = "mi";
				strValue = domain;
				//msup->AddTag(strName, strValue);
				xmlPresentationCreator::WrapIdentifier(msup, domain);

				strName  = "mn";
				strValue = "2";
				msup->AddTag(strName, strValue);
	}

	return mfrac;
}

/*********************************************************************************************
    textCreator
*********************************************************************************************/
string textCreator::Constant(real_t value)
{
	string strValue = toStringFormatted<real_t>(value, -1, 10, false);
	if(value < 0)
		return string("(") + strValue + string(")");
	else
		return strValue;
}
	
string textCreator::Variable(string name, 
							 const vector<string>& domains)
{
	string strDomain;
	string strResult = name;
	RemoveAll(strResult, "&");
	RemoveAll(strResult, ";");

	if(domains.size() > 0)
	{
		strResult += "(";
		for(size_t i = 0; i < domains.size(); i++)
		{
			if(i != 0)
				strResult += ", ";
			strDomain = domains[i];
			RemoveAll(strDomain, "&");
			RemoveAll(strDomain, ";");
			strResult += strDomain;
		}
		strResult += ")";
	}
	return strResult;
}
	
string textCreator::Domain(string name, 
						   string strIndex)
{
	string strResult = name;
	RemoveAll(strResult, "&");
	RemoveAll(strResult, ";");
	strResult += "(";
	strResult += strIndex;
	strResult += ")";
	return strResult;
}
	
string textCreator::TimeDerivative(size_t order, 
								   string name, 
								   const vector<string>& domains,
								   bool bBracketsAroundName)
{
	string strResult, strName, strDomain;
	strResult  = (order == 1 ? "d" : "d2");
	strName = name;
	RemoveAll(strName, "&");
	RemoveAll(strName, ";");
	
	if(bBracketsAroundName)
		strResult += "(";
	strResult += strName;
	if(bBracketsAroundName)
		strResult += ")";

	if(domains.size() > 0)
	{
		strResult += "(";
		for(size_t i = 0; i < domains.size(); i++)
		{
			if(i != 0)
				strResult += ", ";
			strDomain = domains[i];
			RemoveAll(strDomain, "&");
			RemoveAll(strDomain, ";");
			strResult += strDomain;
		}
		strResult += ")";
	}
	strResult += (order == 1 ? "/dt" : "/dt2");
	return strResult;
}
	
string textCreator::PartialDerivative(size_t order, 
									  string name, 
									  string domain, 
									  const vector<string>& domains,
									  bool bBracketsAroundName)
{
	string strResult, strName, strDomain;
	strResult  = (order == 1 ? "d" : "d2");
	strName = name;
	RemoveAll(strName, "&");
	RemoveAll(strName, ";");

	if(bBracketsAroundName)
		strResult += "(";
	strResult += strName;
	if(bBracketsAroundName)
		strResult += ")";

	if(domains.size() > 0)
	{
		strResult += "(";
		for(size_t i = 0; i < domains.size(); i++)
		{
			if(i != 0)
				strResult += ", ";
			strDomain = domains[i];
			RemoveAll(strDomain, "&");
			RemoveAll(strDomain, ";");
			strResult += strDomain;
		}
		strResult += ")";
	}
	strResult += "/d";
	strResult += (order == 1 ? "" : "2");
	strDomain = domain;
	RemoveAll(strDomain, "&");
	RemoveAll(strDomain, ";");
	strResult += strDomain;

	return strResult;
}

/*********************************************************************************************
    latexCreator
*********************************************************************************************/
string latexCreator::Constant(real_t value)
{
	string strResult;
	string strValue = toStringFormatted<real_t>(value, 20, 10, false);
	
	strResult  = "{ ";
	if(value < 0)
	{
		strResult += "\\left( ";
		strResult += strValue;
		strResult += " \\right) ";
	}
	else
	{
		strResult += strValue;
	}
	strResult  += " } ";
	return strResult;
}
	
string latexCreator::Variable(string name, 
							  const vector<string>& domains)
{
	string strResult;
	strResult  = "{ ";
	strResult += name;
	if(domains.size() > 0)
	{
		strResult += "\\left( ";
		strResult += "{ ";
		for(size_t i = 0; i < domains.size(); i++)
		{
			if(i != 0)
				strResult += ", ";
			strResult += domains[i];
		}
		strResult += " } ";
		strResult += "\\right)";
	}
	strResult  += " } ";
	return strResult;
}
	
string latexCreator::Domain(string name, 
							string strIndex)
{
	string strResult;
	strResult  = "{ ";
		strResult += name;
		strResult += " \\left( ";
			strResult += strIndex;
		strResult += " \\right) ";
	strResult  += " } ";
	return strResult;
}
	
string latexCreator::TimeDerivative(size_t order, 
									string name, 
									const vector<string>& domains,
									bool bBracketsAroundName)
{
	string strResult;
	strResult  = "{ "; // Start

		strResult += "{ \\partial ";
		if(order == 2)
			strResult += "^2 ";
	
			strResult += "{ "; // Name
			
			if(bBracketsAroundName)
				strResult += "\\left(";
			strResult += name;
			if(bBracketsAroundName)
				strResult += "\\right)";

			if(domains.size() > 0)
			{
				strResult += "\\left( ";
				strResult += "{ "; // Indexes
				for(size_t i = 0; i < domains.size(); i++)
				{
					if(i != 0)
						strResult += ", ";
					strResult += domains[i];
				}
				strResult += "} "; // Indexes
				strResult += "\\right)";
			}
			strResult += " } "; // Name

		strResult += " } "; // partial

		strResult += "\\over ";

		strResult += "{ \\partial t ";
		if(order == 2)
			strResult += "^2 ";
		strResult += "} ";

	strResult  += "} "; // End
	return strResult;
}
	
string latexCreator::PartialDerivative(size_t order, 
									   string name, 
									   string domain, 
									   const vector<string>& domains,
									   bool bBracketsAroundName)
{
	string strResult;
	strResult  = "{ "; // Start

		strResult += "{ \\partial ";
		if(order == 2)
			strResult += "^2 ";
	
			strResult += "{ "; // Name

			if(bBracketsAroundName)
				strResult += "\\left(";
			strResult += name;
			if(bBracketsAroundName)
				strResult += "\\right)";

			if(domains.size() > 0)
			{
				strResult += "\\left( ";
				strResult += "{ "; // Indexes
				for(size_t i = 0; i < domains.size(); i++)
				{
					if(i != 0)
						strResult += ", ";
					strResult += domains[i];
				}
				strResult += "} "; // Indexes
				strResult += "\\right)";
			}
			strResult += " } "; // Name

		strResult += " } "; // partial

		strResult += "\\over ";

		strResult += "{ \\partial ";
		strResult += domain;
		strResult += (order == 1 ? "" : " ^2");
		strResult += " } ";

	strResult  += "} "; // End
	return strResult;
}



}
}

