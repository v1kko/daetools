#ifndef DAE_DOCSTRINGS_H
#define DAE_DOCSTRINGS_H

/******************************************************************
                            daeModel
******************************************************************/
/*
const char* DOCSTR_Template = ". \n\n"
    "*Arguments:* \n"
    " -  (string): . \n"
    " -  (float): . \n"
    " -  (object): . \n"
    "*Returns:* \n"
    "    object \n\n"
    "*Raises:* \n"
    "    RuntimeError \n";

Code snippet:
    "    documentation:: \n\n"
    "        def any(iterable): \n"
    "            for element in iterable: \n"
    "                if element: \n"
    "                    return True \n"
    "            return False \n\n";
*/

const char* DOCSTR_daeModel = "daeModel";
const char* DOCSTR_CreateEquation = "Creates an equation. \n\n"
    "*Arguments:* \n"
    " - name (string): Name of the equation. \n"
    " - description (string, optional, default = ''): Description of the equation. \n"
    " - scaling (float, optional, default = 1.0): Scaling of the equation. \n"
    "*Returns:* \n"
    "    daeEquation object \n\n"
    "*Raises:* \n"
    "    RuntimeError \n\n"
    "*Examples:* \n"
    "    documentation:: \n\n"
    "        def any(iterable): \n"
    "            for element in iterable: \n"
    "                if element: \n"
    "                    return True \n"
    "            return False \n\n";




#endif
