<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <link rel="stylesheet" href="resources/daetools-fmi.css" type="text/css" media="all" />
        <title>
            DAE Tools FMU: <xsl:value-of select="fmiModelDescription/@modelName"/>
        </title>

      </head>
      <body>

        <div id="wrap">
        <div id="content-wrap">
        <div id="content">
        <div id="main">
        
          <div class="post">

            <h1 style="text-align: center;">
                Model: <xsl:value-of select="fmiModelDescription/@modelName"/> 
            </h1>

            <pre>
                <xsl:value-of select="fmiModelDescription/@description"/>
            </pre>
            
            <xsl:apply-templates select="fmiModelDescription/DefaultExperiment"/>
            <xsl:apply-templates select="fmiModelDescription/CoSimulation"/>
            <xsl:apply-templates select="fmiModelDescription/UnitDefinitions"/>
            <xsl:apply-templates select="fmiModelDescription/TypeDefinitions"/>
            <xsl:apply-templates select="fmiModelDescription/LogCategories"/>
            <xsl:apply-templates select="fmiModelDescription/ModelVariables"/>
            <xsl:apply-templates select="fmiModelDescription/ModelStructure"/>
          </div>

        </div> <!-- main -->

      </div> <!-- wrap -->

      </div> <!-- content -->
      </div> <!-- content-wrap -->

      </body>
    </html>
  </xsl:template>

  <xsl:template match="CoSimulation">
    <div>
      <h2>CoSimulation</h2>
      <table class="width100pc">
        <thead>
          <tr>
            <th align="left">Property</th>
            <th align="left">Value</th>
          </tr>
        </thead>
        <tbody>
            <tr>
              <td align="left">modelIdentifier</td>
              <td align="left">
                  <xsl:value-of select="@modelIdentifier"/>
              </td>
            </tr>
            <tr>
              <td align="left">needsExecutionTool</td>
              <td align="left">
                <xsl:value-of select="@needsExecutionTool"/>
              </td>
            </tr>
            <tr>
              <td align="left">canHandleVariableCommunicationStepSize</td>
              <td align="left">
                <xsl:value-of select="@canHandleVariableCommunicationStepSize"/>
              </td>
            </tr>
            <tr>
              <td align="left">canInterpolateInputs</td>
              <td align="left">
                <xsl:value-of select="@canInterpolateInputs"/>
              </td>
            </tr>
            <tr>
              <td align="left">maxOutputDerivativeOrder</td>
              <td align="left">
                <xsl:value-of select="@maxOutputDerivativeOrder"/>
              </td>
            </tr>
            <tr>
              <td align="left">canRunAsynchronuously</td>
              <td align="left">
                <xsl:value-of select="@canRunAsynchronuously"/>
              </td>
            </tr>
            <tr>
              <td align="left">canBeInstantiatedOnlyOncePerProcess</td>
              <td align="left">
                <xsl:value-of select="@canBeInstantiatedOnlyOncePerProcess"/>
              </td>
            </tr>
            <tr>
              <td align="left">canNotUseMemoryManagementFunctions</td>
              <td align="left">
                <xsl:value-of select="@canNotUseMemoryManagementFunctions"/>
              </td>
            </tr>
            <tr>
              <td align="left">canGetAndSetFMUstate</td>
              <td align="left">
                <xsl:value-of select="@canGetAndSetFMUstate"/>
              </td>
            </tr>
            <tr>
              <td align="left">canSerializeFMUstate</td>
              <td align="left">
                <xsl:value-of select="@canSerializeFMUstate"/>
              </td>
            </tr>

        </tbody>
      </table>

    </div>
  </xsl:template>

  <xsl:template match="UnitDefinitions">
    <div>
      <xsl:if test="count(Unit) > 0">

      <h2>Unit Definitions</h2>

      <table class="width100pc">
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Exponents</th>
            <th align="left">Factor</th>
            <th align="left">Offset</th>
          </tr>
        </thead>
        <tbody>
            <xsl:for-each select="Unit">
              <tr>
                <td align="left">
                    <xsl:value-of select="@name"/>
                </td>
                <xsl:variable name="BaseUnit" select="./*[1]" />
                <td> 
                    <xsl:for-each select="$BaseUnit/@*">
                        <xsl:if test="name() = 'kg'">
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'm'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 's'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'A'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'K'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'mol'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'cd'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                        <xsl:if test="name() = 'rad'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/><sup><xsl:value-of select="."/></sup>
                        </xsl:if>
                    </xsl:for-each>
                </td> 
                <td align="left">
                    <xsl:value-of select="$BaseUnit/@factor"/>
                </td>
                <td align="left">
                    <xsl:value-of select="$BaseUnit/@offset"/>
                </td>
              </tr>
            </xsl:for-each>
        </tbody>
      </table>

      </xsl:if>
    </div>
  </xsl:template>

  <xsl:template match="TypeDefinitions">
    <div>
      <xsl:if test="count(SimpleType) > 0">

      <h2>Type Definitions</h2>

      <table class="width100pc">
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Unit</th>
            <th align="left">Min</th>
            <th align="left">Max</th>
            <th align="left">Nominal</th>
          </tr>
        </thead>
        <tbody>
            <xsl:for-each select="SimpleType">
                <xsl:variable name="child" select="./*[1]" />
                <tr>
                    <td align="left">
                        <xsl:value-of select="@name"/>
                    </td>
                    <td>
                        <xsl:value-of select="$child/@unit"/>
                    </td>
                    <td>
                        <xsl:value-of select="$child/@min"/>
                    </td>
                    <td>
                        <xsl:value-of select="$child/@max"/>
                    </td>
                    <td>
                        <xsl:value-of select="$child/@nominal"/>
                    </td>
                </tr>
            </xsl:for-each>
        </tbody>
      </table>

      </xsl:if>
    </div>
  </xsl:template>
 
  <xsl:template match="LogCategories">
    <div>
      <xsl:if test="count(Category) > 0">

      <h2>Log Categories</h2>
        <p>
          <xsl:for-each select="Category">
              <xsl:value-of select="@name"/>
              <xsl:text> </xsl:text>
          </xsl:for-each>
        </p>
      </xsl:if>
    </div>
  </xsl:template>
  
  <xsl:template match="DefaultExperiment">
    <div>
      <h2>DefaultExperiment</h2>
      <table class="width100pc">
        <thead>
          <tr>
            <th align="left">Property</th>
            <th align="left">Value</th>
          </tr>
        </thead>
        <tbody>
            <tr>
              <td align="left">startTime</td>
              <td align="left">
                  <xsl:value-of select="@startTime"/>
              </td>
            </tr>
            <tr>
              <td align="left">stopTime</td>
              <td align="left">
                <xsl:value-of select="@stopTime"/>
              </td>
            </tr>
            <tr>
              <td align="left">tolerance</td>
              <td align="left">
                <xsl:value-of select="@tolerance"/>
              </td>
            </tr>
        </tbody>
      </table>

    </div>
  </xsl:template>
  
  <xsl:template match="ModelVariables">
    <div>
      <xsl:if test="count(ScalarVariable) > 0">

      <h2>Model Variables</h2>

      <table class="width100pc">
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">ValueReference</th>
            <th align="left">Causality</th>
            <th align="left">Variability</th>
            <th align="left">Initial</th>
            <th align="left">DeclaredType</th>
            <th align="left">Start</th>
            <th align="left">Description</th>
          </tr>
        </thead>
        <tbody>
          <xsl:for-each select="ScalarVariable">
            <tr>
              <td align="left">
                  <xsl:value-of select="@name"/>
              </td>
              <td>
                  <xsl:value-of select="@valueReference"/>
              </td>
              <td>
                  <xsl:value-of select="@causality"/>
              </td>
              <td>
                  <xsl:value-of select="@variability"/>
              </td>
              <td>
                  <xsl:value-of select="@initial"/>
              </td>
              
              <xsl:if test="Real">
                <td>
                  <xsl:value-of select="Real/@declaredType"/>
                </td>
                <td>
                  <xsl:value-of select="Real/@start"/>
                </td>
              </xsl:if>
              <xsl:if test="Integer">
                <td>
                  <xsl:value-of select="Integer/@declaredType"/>
                </td>
                <td>
                  <xsl:value-of select="Integer/@start"/>
                </td>
              </xsl:if>   
              <xsl:if test="Boolean">
                <td></td>
                <td>
                  <xsl:value-of select="Boolean/@start"/>
                </td>
              </xsl:if>              
              <xsl:if test="String">
                <td></td>
                <td>
                  <xsl:value-of select="String/@start"/>
                </td>
              </xsl:if>              
              <xsl:if test="Enumeration">
                <td></td>
                <td>
                  <xsl:value-of select="Enumeration/@start"/>
                </td>
              </xsl:if>              
              
              <td>
                  <xsl:value-of select="@description"/>
              </td>

            </tr>
          </xsl:for-each>
        </tbody>
      </table>

      </xsl:if>
    </div>
  </xsl:template>

  
  <xsl:template match="ModelStructure">
    <xsl:apply-templates select="Outputs"/>
    <xsl:apply-templates select="InitialUnknowns"/>
  </xsl:template>

  <xsl:template match="Outputs">
    <div>
      <xsl:if test="count(Unknown) > 0">
        <h2>Outputs</h2>
        <table class="width100pc">
            <thead>
                <tr>
                    <th align="left">Index</th>
                    <th align="left">Variable</th>
                </tr>
            </thead>
            <tbody>
                <xsl:for-each select="Unknown">
                    <xsl:apply-templates select="."/>
                </xsl:for-each>
            </tbody>
        </table>
      </xsl:if>
    </div>
  </xsl:template>

  <xsl:template match="InitialUnknowns">
    <div>
      <xsl:if test="count(Unknown) > 0">
        <h2>Initial Unknowns</h2>
        <table class="width100pc">
            <thead>
                <tr>
                    <th align="left">Index</th>
                    <th align="left">Variable</th>
                </tr>
            </thead>
            <tbody>
                <xsl:for-each select="Unknown">
                    <xsl:apply-templates select="."/>
                </xsl:for-each>
            </tbody>
        </table>
      </xsl:if>
    </div>
  </xsl:template>

  <xsl:template match="Unknown">
    <tr>
        <td align="left">
            <xsl:value-of select="@index"/>
        </td>
        <td>
            <xsl:variable name="outputIndex" select="@index" />
            <xsl:for-each select="/fmiModelDescription/ModelVariables/ScalarVariable">
                <xsl:if test="position() = $outputIndex">
                    <xsl:value-of select="@name"/>
                </xsl:if>
            </xsl:for-each>
        </td>
    </tr>
  </xsl:template>

</xsl:stylesheet>
