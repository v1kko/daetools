<?xml version="1.0" encoding="utf-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <meta charset="UTF-8"/> 
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <link rel="stylesheet" href="resources/daetools-fmi.css"/>
        <title>
            FMU: <xsl:value-of select="fmiModelDescription/@modelName"/>
        </title>
        <style>
            @media print {
                body.w3-twothird { width: 99%; }
            }
        </style>
      </head>
      <body class="w3-display-container w3-display-topmiddle w3-twothird">
        <div class="w3-section">
          <div class="w3-card w3-round-large w3-padding">
            <div class="w3-container" style="">
                <h3>
                    FMU: <b><xsl:value-of select="fmiModelDescription/@modelName"/></b> 
                </h3>
                <pre class="w3-code w3-small w3-border-0" style="max-height:300px; overflow-y: visible; overflow: auto">
                    <xsl:value-of select="fmiModelDescription/@description"/>
                </pre>
            </div>
            
            <div class="w3-row w3-padding-16">
                <div class="w3-half">
                    <xsl:apply-templates select="fmiModelDescription"/>
                </div>
                <div class="w3-half">
                    <xsl:apply-templates select="fmiModelDescription/CoSimulation"/>
                </div>
            </div>
            
            <div class="w3-row w3-padding-16">
                <div class="w3-half">
                    <xsl:apply-templates select="fmiModelDescription/DefaultExperiment"/>
                </div>
                <div class="w3-half">
                    <xsl:apply-templates select="fmiModelDescription/LogCategories"/>
                </div>
            </div>
            
            <xsl:apply-templates select="fmiModelDescription/UnitDefinitions"/>
            <xsl:apply-templates select="fmiModelDescription/TypeDefinitions"/>
            <xsl:apply-templates select="fmiModelDescription/ModelVariables"/>
            <xsl:apply-templates select="fmiModelDescription/ModelStructure"/>
          </div>
          <br/>
          
        </div>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="fmiModelDescription">
    <div class="w3-container" style="overflow: auto">
      <h3>ModelDescription</h3>
      <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word;">
        <thead>
          <tr class="w3-light-grey">
            <th align="left">Property</th>
            <th align="left">Value</th>
          </tr>
        </thead>
        <tbody>
            <tr>
              <td align="left">fmi Version</td>
              <td align="left">
                  <xsl:value-of select="@fmiVersion"/>
              </td>
            </tr>
            <!-- Already included at the top
            <tr>
              <td align="left">modelName</td>
              <td align="left">
                  <xsl:value-of select="@modelName"/>
              </td>
            </tr>
            -->
            <tr>
              <td align="left">guid</td>
              <td align="left">
                <xsl:value-of select="@guid"/>
              </td>
            </tr>
            <tr>
              <td align="left">author</td>
              <td align="left">
                <xsl:value-of select="@author"/>
              </td>
            </tr>
            <tr>
              <td align="left">version</td>
              <td align="left">
                <xsl:value-of select="@version"/>
              </td>
            </tr>
            <tr>
              <td align="left">copyright</td>
              <td align="left">
                <xsl:value-of select="@copyright"/>
              </td>
            </tr>
            <tr>
              <td align="left">license</td>
              <td align="left">
                <xsl:value-of select="@license"/>
              </td>
            </tr>
            <tr>
              <td align="left">generationTool</td>
              <td align="left">
                <xsl:value-of select="@generationTool"/>
              </td>
            </tr>
            <tr>
              <td align="left">generationDateAndTime</td>
              <td align="left">
                <xsl:value-of select="@generationDateAndTime"/>
              </td>
            </tr>
            <tr>
              <td align="left">variableNamingConvention</td>
              <td align="left">
                <xsl:value-of select="@variableNamingConvention"/>
              </td>
            </tr>
            <tr>
              <td align="left">numberOfEventIndicators</td>
              <td align="left">
                <xsl:value-of select="@numberOfEventIndicators"/>
              </td>
            </tr>
            <!-- Already included at the top
            <tr>
              <td align="left">description</td>
              <td align="left" style="display: block">
                <pre class="w3-code w3-small w3-border-0" style="height:200px; overflow-x:hidden; overflow-y:visible;"><xsl:value-of select="@description"/></pre>
              </td>
            </tr>
            -->

        </tbody>
      </table>

    </div>
    <br/>
  </xsl:template>

  <xsl:template match="CoSimulation">
    <div class="w3-container" style="overflow: auto">
      <h3>CoSimulation</h3>
      <table class="w3-table w3-striped w3-border w3-bordered">
        <thead>
          <tr class="w3-light-grey">
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
    <div class="w3-container w3-padding-16" style="overflow: auto">
      <xsl:if test="count(Unit) > 0">

      <h3>Unit Definitions</h3>

      <table class="w3-table w3-striped w3-border w3-bordered">
        <thead>
          <tr class="w3-light-grey">
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
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'm'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 's'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'A'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'K'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'mol'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'cd'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
                        </xsl:if>
                        <xsl:if test="name() = 'rad'">
                            <xsl:text> </xsl:text>
                            <xsl:value-of select="name()"/>
                            <xsl:if test=". != '1'">
                                <sup><xsl:value-of select="."/></sup>
                            </xsl:if>
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
    <div class="w3-container w3-padding-16" style="overflow: auto">
      <xsl:if test="count(SimpleType) > 0">

        <h3>Type Definitions</h3>

        <table class="w3-table w3-striped w3-border w3-bordered">
            <thead>
            <tr class="w3-light-grey">
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
    <div class="w3-container" style="overflow: auto">
      <xsl:if test="count(Category) > 0">

      <h3>Log Categories</h3>
        <table class="w3-table w3-striped w3-border w3-bordered">
            <thead>
                <tr class="w3-light-grey">
                    <th align="left">Category</th>
                </tr>
            </thead>
            <tbody>
                <xsl:for-each select="Category">
                    <tr>
                        <td><xsl:value-of select="@name"/></td>
                    </tr>
                </xsl:for-each>
            </tbody>
        </table>
      </xsl:if>
    </div>
  </xsl:template>
  
  <xsl:template match="DefaultExperiment">
    <div class="w3-container" style="overflow: auto">
      <h3>DefaultExperiment</h3>
      <table class="w3-table w3-striped w3-border w3-bordered">
        <thead>
          <tr class="w3-light-grey">
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
    <div class="w3-container w3-padding-16" style="overflow: auto">
      <xsl:if test="count(ScalarVariable) > 0">

      <h3>Model Variables</h3>

      <table class="w3-table w3-striped w3-border w3-bordered">
        <thead>
          <tr class="w3-light-grey">
            <th align="left">Index</th>
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
                  <xsl:value-of select="position()"/>
              </td>
               
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
    <div class="w3-row w3-padding-16" style="overflow: auto">
      <div class="w3-half">
        <xsl:apply-templates select="Outputs"/>
      </div>
      <div class="w3-half">
        <xsl:apply-templates select="InitialUnknowns"/>
      </div>
    </div>
  </xsl:template>

  <xsl:template match="Outputs">
    <div class="w3-container" style="overflow: auto">
      <xsl:if test="count(Unknown) > 0">
        <h3>Outputs</h3>
        <table class="w3-table w3-striped w3-border w3-bordered">
            <thead>
                <tr class="w3-light-grey">
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
    <div class="w3-container" style="overflow: auto">
      <xsl:if test="count(Unknown) > 0">
        <h3>Initial Unknowns</h3>
        <table class="w3-table w3-striped w3-border w3-bordered">
            <thead>
                <tr class="w3-light-grey">
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
