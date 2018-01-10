<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <link rel="stylesheet" href="dae-tools.css" type="text/css" media="all" />
        <link rel="shortcut icon" href="http://www.daetools.com/favicon.ico" type="image/x-icon"/>
        <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                jax: ["input/TeX", "output/HTML-CSS"],
                tex2jax: { inlineMath: [ ['$','$'] ] },
                displayAlign: "left",
                CommonHTML: { linebreaks: { automatic: false}, 
                              preferredFont: null, 
                              mtextFontInherit: true, 
                              styles: { '.MathJax_Display': { "margin": 0 } }
                            },
                "HTML-CSS": { linebreaks: { automatic: false}, 
                              preferredFont: null, 
                              mtextFontInherit: true,
                              styles: { '.MathJax_Display': { "margin": 0 } }
                            },
                       SVG: { linebreaks: { automatic: false}, 
                              preferredFont: null, 
                              mtextFontInherit: true 
                            }
            });
        </script>
        <script type="text/javascript" async="true" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
        <title>
            DAE Tools (www.daetools.com) Model Report: <xsl:value-of select="Model/Name"/>
        </title>
        <style>
            @media print {
                body.w3-dae-width { width: 99%; }
            }
            @media screen {
                 body.w3-dae-width {
                    width: 1000px; 
                    max-width: 90% }
            }
            div.w3-left-indent       {padding-left: 30px;}
            div.w3-bottom-indent     {padding-bottom: 10px;}
            div.w3-eqn-bottom-indent {padding-bottom: 0.2em;}
            div.w3-top-bottom-indent {padding-top: 10px; padding-bottom: 10px;}
        </style>
      </head>
      <body class="w3-display-container w3-display-topmiddle w3-dae-width">
        <div>
            <p>
                <img style="height: 1.2em; display: inline" src="http://www.daetools.com/img/[d][a][e]_Tools_project.png" alt=""/>
                <xsl:text> </xsl:text>
                <a href="http://www.daetools.com"><b>DAE Tools Project</b></a>
            </p>        
        
            <div class="w3-card w3-round-large w3-padding">
            
                <div class="w3-container">
                    <h2 style="text-align: center;">
                        Model: <xsl:copy-of select="Model/Name"/> 
                    </h2>
                </div>

                <xsl:if test="Model/Description != ''">
                    <div class="w3-container">
                        <b>Description:</b>
                        <pre class="w3-code w3-small w3-border-0" style="max-height:200px; overflow:auto; overflow-y: visible;">
                            <xsl:value-of select="Model/Description"/>
                        </pre>
                    </div>
                </xsl:if>

                <xsl:apply-templates select="Model/Components"/>
                <xsl:apply-templates select="Model/Ports"/>
                <xsl:apply-templates select="Model/EventPorts"/>
                <xsl:apply-templates select="Model/PortConnections"/>
                <xsl:apply-templates select="Model/EventPortConnections"/>
                <xsl:apply-templates select="Model/Domains"/>
                <xsl:apply-templates select="Model/Parameters"/>
                <xsl:apply-templates select="Model/Variables"/>
                
                <xsl:if test="count(Model/Equations/Object) > 0">
                    <div class="w3-container">
                        <h3><a name="Equations"></a>Equations</h3>
                        <xsl:apply-templates select="Model/Equations"/>
                    </div>
                </xsl:if>

                <xsl:if test="count(Model/STNs/Object) > 0">
                    <div class="w3-container">
                        <h3> <a name="STNs"></a>State Transition Networks</h3>
                        <xsl:apply-templates select="Model/STNs"/>
                    </div>
                </xsl:if>

                <xsl:if test="count(Model/OnEventActions/Object) > 0">
                    <div class="w3-container">
                        <h3><a name="OnEventActions"></a>OnEventActions</h3>
                        <xsl:apply-templates select="Model/OnEventActions"/>
                    </div>
                </xsl:if>
                <br/>

            </div>
            <br/>
        </div>
      </body>
    </html>
  </xsl:template>

  
  <xsl:template match="Domains">
    <xsl:if test="count(Object) > 0">
      <div class="w3-container w3-section">

        <h3> <a name="Domains"></a>Domains</h3>

        <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
          <thead>
            <tr class="w3-light-grey">
              <th>Name</th>
              <th>Type</th>
              <th>Units</th>
              <th>No. Points</th>
              <th>Lower Bound</th>
              <th>Upper Bound</th>
              <th>Description</th>
            </tr>
          </thead>
          <colgroup>
            <col style="width:15%"/>
            <col style="width:10%"/>
            <col style="width:10%"/>
            <col style="width:10%"/>
            <col style="width:15%"/>
            <col style="width:15%"/>
            <col style="width:25%"/>
          </colgroup>
          <tbody>
            <xsl:for-each select="Object">
              <tr>
                <td>
                    <xsl:attribute name="name">
                      <xsl:value-of select="@ID"/>
                    </xsl:attribute>
                    <!-- Add $...$ to display it as inline math? -->
                    <xsl:value-of select="Name"/>
                </td>

                <td>
                    <xsl:if test="Type = 'eStructuredGrid'">
                        Structured Grid
                    </xsl:if>
                    <xsl:if test="Type = 'eUnstructuredGrid'">
                        Unstructured Grid
                    </xsl:if>
                    <xsl:if test="Type = 'eArray'">
                        Array
                    </xsl:if>
                </td>

                <td>
                    <xsl:apply-templates select="Units"/>
                </td>
                
                <td>
                  <xsl:value-of select="number(NumberOfPoints)"/>
                </td>

                <td>
                  <xsl:value-of select="number(LowerBound)"/>
                </td>

                <td>
                  <xsl:value-of select="number(UpperBound)"/>
                </td>

                <td>
                  <xsl:value-of select="Description"/>
                </td>

              </tr>
            </xsl:for-each>
          </tbody>
        </table>

      </div>
    </xsl:if>
  </xsl:template>

  
  <xsl:template match="Units">
    <xsl:for-each select="./*">
      <xsl:choose>
        <!-- If exponent is 0 do nothing --> 
        <xsl:when test="number(.) = 0">
        </xsl:when>
        <!-- If exponent is 1 add only the unit i.e. kg --> 
        <xsl:when test="number(.) = 1">
            <xsl:value-of select="name()"/>
            <xsl:text> </xsl:text>
        </xsl:when>
        <!-- Otherwise add name ^ exponent i.e. kg^2 --> 
        <xsl:otherwise>
            <xsl:value-of select="name()"/>
            <sup><xsl:value-of select="number(.)"/></sup>
            <xsl:text> </xsl:text>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  </xsl:template>

  
  <xsl:template match="DomainRefs">
    <xsl:for-each select="./ObjectRef">

      <a style="text-decoration: none">
        <xsl:attribute name="href">
          #<xsl:value-of select="@ID"/>
        </xsl:attribute>
        <xsl:value-of select="."/>
      </a>

      <xsl:if test="not(position() = last())">
        <xsl:text>, </xsl:text>
      </xsl:if>

    </xsl:for-each>
  </xsl:template>

  
  <xsl:template match="Parameters">
    <xsl:if test="count(Object) > 0">
      <div class="w3-container w3-section">

      <h3> <a name="Parameters"></a>Parameters</h3>

      <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
        <thead>
          <tr class="w3-light-grey">
            <th align="left">Name</th>
            <th align="left">Units</th>
            <th align="left">Domains</th>
            <th align="left">Value(s)</th>
            <th align="left">Description</th>
          </tr>
        </thead>
        <colgroup>
            <col style="width:15%"/>
            <col style="width:15%"/>
            <col style="width:15%"/>
            <col style="width:30%"/>
            <col style="width:25%"/>
        </colgroup>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td align="left">
                <!-- Add $...$ to display it as an inline math -->
                <xsl:value-of select="Name"/>
              </td>

              <td align="left">
                <xsl:apply-templates select="Units"/>
              </td>

              <td align="left">
                <xsl:apply-templates select="DomainRefs"/>
              </td>

              <td>
                  <xsl:if test="count(Values/Item) > 1">
                    <xsl:text>[</xsl:text>
                  </xsl:if>
                  
                  <xsl:for-each select="Values/Item">
                    <xsl:if test="position() != 1">
                      <xsl:text>, </xsl:text>
                    </xsl:if>
                    
                    <xsl:value-of select="number(.)" />
                  </xsl:for-each>
                  
                  <xsl:if test="count(Values/Item) > 1">
                    <xsl:text>]</xsl:text>
                  </xsl:if>
              </td>
              
              <td>
                  <xsl:value-of select="Description"/>
              </td>

            </tr>
          </xsl:for-each>
        </tbody>
      </table>

      </div>
    </xsl:if>
  </xsl:template>

  
  <xsl:template match="Variables">
    <xsl:if test="count(Object) > 0">
     <div class="w3-container w3-section">

      <h3> <a name="Variables"></a>Variables</h3>

      <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
        <thead>
          <tr class="w3-light-grey">
            <th align="left">Name</th>
            <th align="left">Type</th>
            <th align="left">Domains</th>
            <th align="left">Overall Index</th>
            <th align="left">Description</th>
          </tr>
        </thead>
        <colgroup>
            <col style="width:15%"/>
            <col style="width:15%"/>
            <col style="width:15%"/>
            <col style="width:30%"/>
            <col style="width:25%"/>
        </colgroup>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td align="left">
                <!-- Add $...$ to display it as inline math? -->
                <xsl:value-of select="Name"/>
              </td>

              <td align="left">
                <xsl:value-of select="VariableType"/>
              </td>

              <td align="left">
                <xsl:apply-templates select="DomainRefs"/>
              </td>

              <td>
                  <xsl:value-of select="number(OverallIndex)"/>
              </td>

              <td>
                  <xsl:value-of select="Description"/>
              </td>

            </tr>
          </xsl:for-each>
        </tbody>
      </table>

     </div>
    </xsl:if>
  </xsl:template>

  
  <xsl:template match="Components">
      <xsl:if test="count(Object) > 0">
         <div class="w3-container w3-section">
            <h3> <a name="Components"></a>Components</h3>
            <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
                <thead>
                    <tr class="w3-light-grey">
                        <th align="left">Name</th>
                        <th align="left">Description</th>
                    </tr>
                </thead>
                <colgroup>
                    <col style="width:40%"/>
                    <col style="width:60%"/>
                </colgroup>
                <tbody>
                    <xsl:for-each select="Object">
                        <tr>
                            <td align="left">
                                <!-- Add $...$ to display it as inline math?? -->
                                <xsl:value-of select="Name"/>
                            </td>
                            <td align="left">
                                <xsl:value-of select="Description"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </tbody>
            </table>
         </div>
       </xsl:if>
  </xsl:template>


  <xsl:template match="Ports">
     <xsl:if test="count(Object) > 0">
        <div class="w3-container w3-section">
            <h3> <a name="Ports"></a>Ports</h3>
            <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
                <thead>
                    <tr class="w3-light-grey">
                        <th align="left">Name</th>
                        <th align="left">Type</th>
                        <th align="left">Description</th>
                    </tr>
                </thead>
                <colgroup>
                    <col style="width:30%"/>
                    <col style="width:30%"/>
                    <col style="width:40%"/>
                </colgroup>
                <tbody>
                    <xsl:for-each select="Object">
                        <tr>
                            <td align="left">
                                <!-- Add $...$ to display it as inline math?? -->
                                <xsl:value-of select="Name"/>
                            </td>

                            <td align="left">
                                <xsl:value-of select="PortType"/>
                            </td>

                            <td>     
                                <xsl:value-of select="Description"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </tbody>
            </table>

       </div>
     </xsl:if>
  </xsl:template>

  
  <xsl:template match="EventPorts">
     <xsl:if test="count(Object) > 0">
        <div class="w3-container w3-section">
            <h3> <a name="EventPorts"></a>EventPorts</h3>
            <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
                <thead>
                    <tr class="w3-light-grey">
                        <th align="left">Name</th>
                        <th align="left">Type</th>
                        <th align="left">Description</th>
                    </tr>
                </thead>
                <colgroup>
                    <col style="width:30%"/>
                    <col style="width:30%"/>
                    <col style="width:40%"/>
                </colgroup>
                <tbody>
                    <xsl:for-each select="Object">
                        <tr>
                            <td align="left">
                                <!-- Add $...$ to display it as inline math?? -->
                                <xsl:value-of select="Name"/>
                            </td>

                            <td align="left">
                                <xsl:value-of select="PortType"/>
                            </td>

                            <td>
                                <xsl:value-of select="Description"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </tbody>
            </table>

       </div>
    </xsl:if>
  </xsl:template>

  
  <xsl:template match="PortConnections">
     <xsl:if test="count(Object) > 0">
        <div class="w3-container w3-section">
            <h3> <a name="PortConnections"></a>Port Connections</h3>
            <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
                <thead>
                    <tr class="w3-light-grey">
                        <th align="left">Port 1</th>
                        <th align="left">Port 2</th>
                        <th align="left">Description</th>
                    </tr>
                </thead>
                <colgroup>
                    <col style="width:30%"/>
                    <col style="width:30%"/>
                    <col style="width:40%"/>
                </colgroup>
                <tbody>
                    <xsl:for-each select="Object">
                        <tr>
                            <td align="left">
                                <!-- Add $...$ to display it as inline math? -->
                                <xsl:value-of select="PortFrom"/>
                            </td>

                            <td align="left">
                                <!-- Add $...$ to display it as inline math? -->
                                <xsl:value-of select="PortTo"/>
                            </td>

                            <td>     
                                <xsl:value-of select="Description"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </tbody>
            </table>

       </div>
     </xsl:if>
  </xsl:template>

  
  <xsl:template match="EventPortConnections">
    <xsl:if test="count(Object) > 0">
        <div class="w3-container w3-section">
            <h3> <a name="EventPortConnections"></a>Event-Port Connections</h3>
            <table class="w3-table w3-striped w3-border w3-bordered" style="word-wrap: break-word; table-layout:fixed">
                <thead>
                    <tr class="w3-light-grey">
                        <th align="left">Port 1</th>
                        <th align="left">Port 2</th>
                        <th align="left">Description</th>
                    </tr>
                </thead>
                <colgroup>
                    <col style="width:30%"/>
                    <col style="width:30%"/>
                    <col style="width:40%"/>
                </colgroup>
                <tbody>
                    <xsl:for-each select="Object">
                        <tr>
                            <td align="left">
                                <!-- Add $...$ to display it as inline math? -->
                                <xsl:value-of select="PortFrom"/>
                            </td>

                            <td align="left">
                                <!-- Add $...$ to display it as inline math? -->
                                <xsl:value-of select="PortTo"/>
                            </td>

                            <td>
                                <xsl:value-of select="Description"/>
                            </td>
                        </tr>
                    </xsl:for-each>
                </tbody>
            </table>

       </div>
    </xsl:if>
  </xsl:template>

  
  <xsl:template match="Equations">
      <xsl:if test="count(Object) > 0">
          <xsl:for-each select="Object">
              <div class="w3-eqn-bottom-indent">
                  <div class="w3-eqn-bottom-indent">
                    <!-- Add $...$ to display it as inline math? -->
                    <xsl:value-of select="Name"/> 
                    <xsl:text>:</xsl:text>
                  </div>
                  
                  <div class="w3-left-indent" style="overflow: auto; overflow-y: hidden">
                    <xsl:copy-of select="Residual"/>
                    <!--
                    <xsl:if test="Description != ''">
                        <i> <xsl:value-of select="Description"/> </i>
                    </xsl:if>
                    -->                                
                  </div>
                  
                  <div class="w3-small w3-left-indent">
                      <p>Expanded into:</p>
                      <xsl:apply-templates select="EquationExecutionInfos"/>
                  </div>
             </div>
          </xsl:for-each>
      </xsl:if>
  </xsl:template>

  
  <xsl:template match="EquationExecutionInfos">
      <xsl:if test="count(Object) > 0">

          <xsl:for-each select="Object">
             <div>
                <div class="w3-left-indent" style="overflow: auto; overflow-y: hidden">
                    <xsl:copy-of select="EquationEvaluationNode"/>
                
                    <div class="w3-text-grey">
                        Equation is: 
                        <xsl:if test="IsLinear = 'True'">
                            linear
                        </xsl:if>
                        <xsl:if test="IsLinear = 'False'">
                            non-linear
                        </xsl:if>
                    </div>
                </div>
             </div>
          </xsl:for-each>

      </xsl:if>
  </xsl:template>

  
  <xsl:template match="STNs">
    <div>    
      <xsl:for-each select="Object">

        <xsl:if test="Type = 'eIF'">
          <div class="w3-bottom-indent">
                <xsl:for-each select="States/Object">
                    <div>
                        <b>
                        <xsl:if test="position() = 1">
                            If <xsl:apply-templates select="OnConditionActions/Object/Condition"/>:
                        </xsl:if>

                        <xsl:if test="position() > 1 and position() != last()">
                            Else if <xsl:apply-templates select="OnConditionActions/Object/Condition"/>:
                        </xsl:if>

                        <xsl:if test="position() = last()">
                            Else:
                        </xsl:if>
                        </b>

                        <!-- OnConditionActions and OnEventActions do not make much sense within the IF/ELSE_IF/ELSE blocks-->
                        <div class="w3-left-indent">
                            <xsl:apply-templates select="Equations"/>
                        </div>
                        <div class="w3-left-indent w3-top-bottom-indent">
                            <xsl:apply-templates select="STNs"/>
                        </div>
                        
                        <xsl:if test="position() = last()">
                            <div>
                                <b>End if</b>
                            </div>
                        </xsl:if>
                    </div>
                </xsl:for-each>
          </div>
        </xsl:if>

        <xsl:if test="Type = 'eSTN'">
            <div class="w3-bottom-indent">
                <!-- Add $...$ to display it as inline math? -->
                STN  <xsl:value-of select="Name"/> 
                <xsl:if test="Description != ''">
                    <i><span class="w3-text-grey">(<xsl:value-of select="Description"/>) </span></i>
                </xsl:if>
                :
                <xsl:for-each select="States/Object">
                    <div class="w3-left-indent">
                        state <i> <xsl:value-of select="Name"/>: </i>
                        <div class="w3-left-indent">
                            <xsl:apply-templates select="Equations"/>
                        </div>
                        <div class="w3-left-indent">
                            <xsl:apply-templates select="OnConditionActions"/>
                        </div>
                        <div class="w3-left-indent">
                            <xsl:apply-templates select="OnEventActions"/>
                        </div>
                        <div class="w3-left-indent w3-top-bottom-indent">
                            <xsl:apply-templates select="STNs"/>
                        </div>
                    </div>
                </xsl:for-each>
                
                End STN
            </div>
        </xsl:if>

      </xsl:for-each>
    </div>
  </xsl:template>

  
   <xsl:template match="Actions">
        <div>
            <xsl:if test="count(Object) > 0">
                <div class="w3-left-indent">
                    <xsl:for-each select="Object">
                        <div>
                            <xsl:if test="Type = 'eChangeState'">
                                Switch <i><xsl:value-of select="STN"/></i> to <i><xsl:value-of select="StateTo"/></i>
                            </xsl:if>
                            <xsl:if test="Type = 'eSendEvent'">
                                Trigger the event on <i><xsl:value-of select="SendEventPort"/></i> with the data: <i><xsl:copy-of select="Expression"/></i>
                            </xsl:if>
                            <xsl:if test="Type = 'eReAssignOrReInitializeVariable'">
                                Set <i><xsl:value-of select="Variable"/></i> to <i><xsl:copy-of select="Expression"/></i>
                            </xsl:if>
                        </div>
                    </xsl:for-each>
                </div>
            </xsl:if>
        </div>
  </xsl:template>

  
  <xsl:template match="OnConditionActions">
    <div>
      <xsl:for-each select="Object">
        <div>
            When the condition <xsl:apply-templates select="Condition"/> is satisfied:
            <xsl:apply-templates select="Actions"/>
        </div>
      </xsl:for-each>
    </div>
  </xsl:template>

  
  <xsl:template match="OnEventActions">
    <div>
      <xsl:for-each select="Object">
        <div>
            On event received on <i><xsl:value-of select="EventPort"/></i>:
            <xsl:apply-templates select="Actions"/>
        </div>
      </xsl:for-each>
    </div>
  </xsl:template>

  
  <xsl:template match="Condition">
    <xsl:copy-of select="Expression"/>
  </xsl:template>

</xsl:stylesheet>
