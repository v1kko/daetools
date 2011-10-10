#ifndef PARSER_OBJECTS_H
#define PARSER_OBJECTS_H

#include <boost/config.hpp>
#include <boost/operators.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/variant.hpp>
#include <boost/any.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <stdexcept>
#include <math.h>

namespace parser_objects
{
enum eUnaryOperation
{
    opUnaryMinus,
    opUnaryPlus
};

enum eBinaryOperation
{
    opMinus,
    opPlus,
    opMulti,
    opDivide,
    opPower
};

template<typename TYPE>
class Node
{
public:
	typedef typename boost::variant< boost::function<TYPE ()>,
	                                 boost::function<TYPE (TYPE)>,
 	                                 boost::function<TYPE (TYPE, TYPE)>,
	                                 boost::function<TYPE (TYPE, TYPE, TYPE)>
	                               >			       FUNCTION;
    typedef typename std::map<std::string, TYPE>       mapIdentifiers;
    typedef typename std::map<std::string, FUNCTION>   mapFunctions;
    typedef typename boost::shared_ptr< Node<TYPE> >   node_ptr;

    virtual ~Node(void) {}
    
    virtual TYPE        evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const = 0;
    virtual std::string toString(void) const = 0;
};

template<typename TYPE>
class ConstantNode : public Node<TYPE>
{
    typedef typename Node<TYPE>::mapIdentifiers mapIdentifiers;
    typedef typename Node<TYPE>::mapFunctions   mapFunctions;
    typedef typename Node<TYPE>::node_ptr       node_ptr;
public:
    ConstantNode(double val)
    {
        value = val;
    }
    
    TYPE evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const
    {
        return TYPE(value);
    }
    
    std::string toString(void) const
    {
        return boost::lexical_cast<std::string>(value); 
    }

protected:
    double value;
};

template<typename TYPE>
class IdentifierNode : public Node<TYPE>
{
    typedef typename Node<TYPE>::mapIdentifiers     mapIdentifiers;
    typedef typename Node<TYPE>::mapFunctions       mapFunctions;
    typedef typename Node<TYPE>::node_ptr           node_ptr;
	typedef typename mapIdentifiers::const_iterator id_iterator;
	typedef typename mapFunctions::const_iterator   fn_iterator;
public:
    IdentifierNode(std::string id)
    {
        identifier = id;
    }
    
    TYPE evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const
    {       
        id_iterator iter = ids.find(identifier);
        if(iter == ids.end())
            throw std::runtime_error("");
        return iter->second;
    }
    
    std::string toString(void) const
    {
        return identifier;
    }

protected:
    std::string identifier;
};

template<typename TYPE>
class UnaryNode : public Node<TYPE>
{
    typedef typename Node<TYPE>::mapIdentifiers mapIdentifiers;
    typedef typename Node<TYPE>::mapFunctions   mapFunctions;
    typedef typename Node<TYPE>::node_ptr       node_ptr;
public:
    UnaryNode(node_ptr n, eUnaryOperation o)
    {
        node  = n;
        op    = o;
    }
    
    TYPE evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const
    {
        TYPE n = node->evaluate(ids, fns);
        
        if(op == opUnaryPlus)
            return n;
        else if(op == opUnaryMinus)
            return -n;
        else
            throw std::runtime_error("");
		
		return TYPE();
    }

    std::string toString(void) const
    {
        boost::format fmt;
        std::string format;
        std::string n = node->toString();
        
        if(op == opUnaryPlus)
            format = "+%1%";
        else if(op == opUnaryMinus)
            format = "-%1%";
        else
            throw std::runtime_error("");
        
        fmt.parse(format);
        return (fmt % n).str();     
    }
    
protected:
    node_ptr        node;
    eUnaryOperation op;
};

template<typename TYPE>
class BinaryNode : public Node<TYPE>
{
    typedef typename Node<TYPE>::mapIdentifiers     mapIdentifiers;
    typedef typename Node<TYPE>::mapFunctions       mapFunctions;
    typedef typename Node<TYPE>::node_ptr           node_ptr;
public:
    BinaryNode(node_ptr l, node_ptr r, eBinaryOperation o)
    {
        left  = l;
        right = r;
        op    = o;
    }
    
    TYPE evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const
    {
        TYPE l = left->evaluate(ids, fns);
        TYPE r = right->evaluate(ids, fns);
        
        if(op == opPlus)
            return l + r;
        else if(op == opMinus)
            return l - r;
        else if(op == opMulti)
            return l * r;
        else if(op == opDivide)
            return l / r;
        else if(op == opPower)
            return pow(l, r);
        else
            throw std::runtime_error("");
		
		return TYPE();
    }

    std::string toString(void) const
    {
        boost::format fmt;
        std::string format;
        std::string l = left->toString();
        std::string r = right->toString();
        
        if(op == opPlus)
            format = "%1% + %2%";
        else if(op == opMinus)
            format = "%1% - %2%";
        else if(op == opMulti)
            format = "%1% * %2%";
        else if(op == opDivide)
            format = "%1% / %2%";
        else if(op == opPower)
            format = "%1% ^ %2%";
        else
            throw std::runtime_error("");
        
        fmt.parse(format);
        return (fmt % l % r).str();     
    }
    
protected:
    node_ptr         left;
    node_ptr         right;
    eBinaryOperation op;
};

template<typename TYPE>
class FunctionNode : public Node<TYPE>
{
    typedef typename boost::function<TYPE ()>                 FUNCTION_0;
    typedef typename boost::function<TYPE (TYPE)>             FUNCTION_1;
    typedef typename boost::function<TYPE (TYPE, TYPE)>       FUNCTION_2;
    typedef typename boost::function<TYPE (TYPE, TYPE, TYPE)> FUNCTION_3;
    
	typedef typename Node<TYPE>::FUNCTION                     FUNCTION;
    typedef typename Node<TYPE>::mapIdentifiers				  mapIdentifiers;
    typedef typename Node<TYPE>::mapFunctions   			  mapFunctions;
    typedef typename Node<TYPE>::node_ptr       			  node_ptr;
    typedef typename mapIdentifiers::const_iterator           id_iterator;
    typedef typename mapFunctions::const_iterator             fn_iterator;
public:
    FunctionNode(const std::string& fun)
    {
        function  = fun;
    }
    FunctionNode(const std::string& fun, const node_ptr& arg0)
    {
        arguments.push_back(arg0);
        function  = fun;
    }
    FunctionNode(const std::string& fun, const node_ptr& arg0, const node_ptr& arg1)
    {
        arguments.push_back(arg0);
        arguments.push_back(arg1);
        function  = fun;
    }
    FunctionNode(const std::string& fun, const node_ptr& arg0, const node_ptr& arg1, const node_ptr& arg2)
    {
        arguments.push_back(arg0);
        arguments.push_back(arg1);
        arguments.push_back(arg2);
        function  = fun;
    }
    
    TYPE evaluate(const mapIdentifiers& ids, const mapFunctions& fns) const
    {
        fn_iterator iter = fns.find(function);
        if(iter == fns.end())
            throw std::runtime_error("");
        FUNCTION fun = iter->second;
		 
        if(FUNCTION_0* fn = boost::get<FUNCTION_0>(&fun))
		{
			if(arguments.size() != 0)
				throw std::runtime_error("");
			return fn();
        }
        else if(FUNCTION_1* fn = boost::get<FUNCTION_1>(&fun))
		{
			if(arguments.size() != 1)
				throw std::runtime_error("");
			TYPE arg0 = arguments[0]->evaluate(ids, fns);
			return fn(arg0);
        }
        else if(FUNCTION_3* fn = boost::get<FUNCTION_3>(&fun))
		{
			if(arguments.size() != 3)
				throw std::runtime_error("");
			TYPE arg0 = arguments[0]->evaluate(ids, fns);
			TYPE arg1 = arguments[1]->evaluate(ids, fns);
			TYPE arg2 = arguments[2]->evaluate(ids, fns);
			return fn(arg0, arg1, arg2);
        }
        else
            throw std::runtime_error("");
		
		return TYPE();
    }

    std::string toString(void) const
    {
        if(arguments.size() == 1)
		{
			std::string arg0 = arguments[0]->toString();
	        return (boost::format("%1%(%2%)") % function % arg0).str();     
        }
        else
            throw std::runtime_error("");
		
		return std::string();
    }
    
protected:
    std::vector<node_ptr> arguments;
    std::string           function;
};


template<class TYPE>
class Number 
{
public:
    typedef typename boost::shared_ptr< Node<TYPE> > node_ptr;
    
    Number(node_ptr n)
    {
        node = n;
    }

    Number<TYPE> operator+(const Number<TYPE>& val)
    {
        node_ptr n(new BinaryNode<TYPE>(node, val.node, opPlus));
        return Number<TYPE>(n);
    }

    Number<TYPE> operator-(const Number<TYPE>& val)
    {
        node_ptr n(new BinaryNode<TYPE>(node, val.node, opMinus));
        return Number<TYPE>(n);
    }

    Number<TYPE> operator*(const Number<TYPE>& val)
    {
        node_ptr n(new BinaryNode<TYPE>(node, val.node, opMulti));
        return Number<TYPE>(n);
    }

    Number<TYPE> operator/(const Number<TYPE>& val)
    {
        node_ptr n(new BinaryNode<TYPE>(node, val.node, opDivide));
        return Number<TYPE>(n);
    }

    Number<TYPE> operator^(const Number<TYPE>& val)
    {
        node_ptr n(new BinaryNode<TYPE>(node, val.node, opPower));
        return Number<TYPE>(n);
    }
	
public:
    node_ptr node;
};



}

#endif // PARSER_OBJECTS_H
