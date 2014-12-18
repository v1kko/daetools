"""********************************************************************************
                             html_form.py
                 DAE Tools: pyDAE module, www.daetools.com
                 Copyright (C) Dragan Nikolic, 2014
***********************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
********************************************************************************"""

css_styles = """
<style type="text/css">
    body {
        color: #333333;
    }

    html, body {
        height: 100%;
        width: 800px;
        margin: 0 auto;

        color: #555;
        background-color: #FFFFFF;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #333333
    }

    h1 {
        #height: 23px;
        margin: 0px;
        border-bottom: 5px solid #636466;
        color: #636466;
        display: block;
    }

    table {
        border-collapse:collapse;
        border:thin solid;
    }

    tr {
    }

    td {
        border: thin solid;
        padding-left:   10px;
        padding-right:  5px;
        padding-top:    3px;
        padding-bottom: 3px;
    }

    th {
        border: thin solid;
        padding-left:   10px;
        padding-right:  5px;
        padding-top:    7px;
        padding-bottom: 7px;
    }

    thead th, tfoot th, tfoot td {
        background-color: #65944A;
        color: white;
    }

    tfoot td {
        text-align:right
    }

</style>

<style type="text/css">

body
{
    font: 0.9em sans-serif;
}
.error
{
    color: red;
}
form
{
    margin:0;
    padding:0;
}
fieldset
{
    margin:1em 0;
    border:none;
    border-top:1px solid #ccc;
}
/*
textarea
{
    border: 1px solid LightGray;
}
input
{
    border: 1px solid LightGray;
}
input:hover
{
    border: 1px solid Gray;
    background: white;
}
textarea:hover
{
    border: 1px solid Gray;
    background: white;
}
select
{
    color: #202020;
    background-color: White;
    width: 200px;
}
*/

/*
input:focus,textarea:focus
{
    background:#efefef;
    color:#000;
}
*/

form fieldset
{
    margin-bottom: 10px;
}
form legend
{
    padding: 0 2px;
    font-weight: bold;
}
form label
{
    display: inline-block;
    line-height: 2;
    vertical-align: middle;
}
form fieldset ol
{
    margin: 0;
    padding: 0;
}
form fieldset li
{
    list-style: disc;
    padding: 0px;
    margin: 0;
}
form fieldset fieldset
{
    border: none;
    margin: 3px 0 0;
}
form fieldset fieldset legend
{
    padding: 0 0 5px;
    font-weight: normal;
}
form fieldset fieldset label
{
    display: block;
    width: auto;
}
form em
{
    font-weight: bold;
    font-style: normal;
    color: #f00;
}
form label
{
   /* min-width: 10em;*/ /* Width of labels */
}
form fieldset fieldset label
{
    margin-left: 17em; /* Width plus 2 (html space) */
}

.ninemlBoolean
{
    width: 10em; /* Width of input fields */
    float: right;
    clear: both;
}

.ninemlFloat
{
    width: 10em; /* Width of input fields */
    float: right;
    clear: both;
    vertical-align: middle;
}

.ninemlString
{
    width: 10em; /* Width of input fields */
    float: right;
    clear: both;
    vertical-align: middle;
}

.ninemlMultilineString
{
    width: 10em; /* Width of input fields */
    float: right;
    white-space:normal;
    clear: both;
    vertical-align: middle;
}

.ninemlComboBox
{
    width: 10em; /* Width of input fields */
    float: right;
    clear: both;
}

.ninemlCheckBox
{
    float: right;
    clear: both;
    vertical-align: middle;
}

input.invalid
{
    color : red;
    border: 2px solid red;
}

label.invalid
{
    color : red;
}
</style>
"""

html_template = """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
<head>
<link type="text/css" href="daetools-webform.css" rel="stylesheet" media="all" />
%(css_style)s
<body>
<div id="main">
<h1>
    DAE Tools: %(name)s
</h1>

%(content)s
</div>

</body>
</head>
"""

