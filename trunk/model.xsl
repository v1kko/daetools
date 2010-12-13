<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <head>
        <link rel="stylesheet" href="model.css" type="text/css" media="screen,projection" />
      </head>
      <body>

        <div style="position:absolute; left:0px">
          <h1>
            MODEL: <xsl:value-of select="Model/@Class"/>
          </h1>
          <br></br>

          <div style="position:relative; left:20px">
            <h2>Domains</h2>
            <xsl:apply-templates select="Model/Domains"/>
            <br></br>
          </div>

          <div style="position:relative; left:20px">
            <h2>Parameters</h2>
            <xsl:apply-templates select="Model/Parameters"/>
            <br></br>
          </div>

          <div style="position:relative; left:20px">
            <h2>Variables</h2>
            <xsl:apply-templates select="Model/Variables"/>
            <br></br>
          </div>

          <div style="position:relative; left:20px">
            <h2>Ports</h2>
            <xsl:apply-templates select="Model/Ports"/>
            <br></br>
          </div>

          <div style="position:relative; left:20px">
            <h2>Equations</h2>
            <xsl:apply-templates select="Model/Equations"/>
            <br></br>
          </div>

          <div style="position:relative; left:20px">
            <h2>State Transition Networks</h2>
            <xsl:apply-templates select="Model/STNs"/>
            <br></br>
          </div>

        </div>

      </body>
    </html>
  </xsl:template>

  <xsl:template match="Domains">
    <div>
      <xsl:if test="count(Object) > 0">
        <table border="0">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            <xsl:for-each select="Object">
              <tr>
                <td>
                  <a>
                    <xsl:attribute name="name">
                      <xsl:value-of select="@ID"/>
                    </xsl:attribute>
                    <xsl:value-of select="Name"/>
                  </a>
                </td>
                <td>
                  <xsl:value-of select="Type"/>
                </td>
              </tr>
            </xsl:for-each>
          </tbody>
        </table>
      </xsl:if>
    </div>
  </xsl:template>

  <xsl:template match="DomainRefs">
    <xsl:for-each select="./ObjectRef">
      <a>
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
    <div>
      <table border="0" cellpadding="10" cellspacing="10">
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Type</th>
            <th align="left">Domains</th>
          </tr>
        </thead>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td align="left">
                <xsl:value-of select="Name"/>
              </td>
              <td align="left">
                <xsl:value-of select="Type"/>
              </td>
              <td align="left">
                <xsl:apply-templates select="DomainRefs"/>
              </td>
            </tr>
          </xsl:for-each>
        </tbody>
      </table>
    </div>
  </xsl:template>

  <xsl:template match="Variables">
    <div>
      <table>
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Type</th>
            <th align="left">Domains</th>
          </tr>
        </thead>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td align="left">
                <xsl:value-of select="Name"/>
              </td>
              <td align="left">
                <xsl:value-of select="VariableType"/>
              </td>
              <td align="left">
                <xsl:apply-templates select="DomainRefs"/>
              </td>
            </tr>
          </xsl:for-each>
        </tbody>
      </table>
    </div>
  </xsl:template>

  <xsl:template match="Ports">
    <div>

      <!--<xsl:for-each select="Object">

        <xsl:value-of select="Name"/>
        <xsl:text>, </xsl:text>
        <xsl:value-of select="PortType"/>

        <div style="position:relative; left:50px">
          <p>Domains</p>
          <xsl:apply-templates select="Domains"/>
        </div>

        <div style="position:relative; left:50px">
          <p>Parameters</p>
          <xsl:apply-templates select="Parameters"/>
        </div>

        <div style="position:relative; left:50px">
          <p>Variables</p>
          <xsl:apply-templates select="Variables"/>
        </div>

      </xsl:for-each>-->

      <table>
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Type</th>
            <th align="left">Class</th>
            <th align="left">Library</th>
            <th align="left">Version</th>
          </tr>
        </thead>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td align="left" valign="top">
                <xsl:value-of select="Name"/>
              </td>
              <td align="left" valign="top">
                <xsl:value-of select="PortType"/>
              </td>
              <td align="left" valign="top">
                <xsl:apply-templates select="@Class"/>
              </td>
              <td align="left" valign="top">
                <xsl:apply-templates select="@Library"/>
              </td>
              <td align="left" valign="top">
                <xsl:apply-templates select="@Version"/>
              </td>
            </tr>
          </xsl:for-each>
        </tbody>
      </table>
    </div>
  </xsl:template>

  <xsl:template match="Equations">
    <div>
      <table>
        <thead>
          <tr>
            <th align="left">Name</th>
            <th align="left">Equation</th>
            <!--<th align="left">Domains</th>-->
          </tr>
        </thead>
        <tbody>
          <xsl:for-each select="Object">
            <tr>
              <td>
                <xsl:value-of select="Name"/>
              </td>
              <td>
                <xsl:copy-of select="MathML"/>
              </td>
            </tr>
          </xsl:for-each>
        </tbody>
      </table>
    </div>

    <!--<table border="0" cellpadding="10" cellspacing="10">
      <tr>
        -->
    <!--<th align="left">ID</th>-->
    <!--
        <th align="left">Name</th>
        <th align="left">MathML</th>
        <th align="left">Domains</th>
        -->
    <!--<th align="left">Definition</th>
        <th align="left">Evaluation</th>
        <th align="left">Expression</th>-->
    <!--
      </tr>
      <xsl:for-each select="Object">
        <tr>
          -->
    <!--<td align="left">
            <xsl:value-of select="@ID"/>
          </td>-->
    <!--
          <td align="left">
            <xsl:value-of select="Name"/>
          </td>
          <td align="left">
            <xsl:copy-of select="MathML"/>
          </td>
          <td align="left">
            <xsl:apply-templates select="DistributedDomainInfos"/>
          </td>
          -->
    <!--<td align="left">
            <xsl:value-of select="EquationDefinitionMode"/>
          </td>
          <td align="left">
            <xsl:value-of select="EquationEvaluationMode"/>
          </td>
          <td align="left">
            <xsl:value-of select="Expression"/>
          </td>-->
    <!--
        </tr>
      </xsl:for-each>
    </table>-->
  </xsl:template>

  <xsl:template match="DistributedDomainInfos">
    <table border="0" cellpadding="10" cellspacing="10">
      <tr>
        <th align="left">Domain</th>
        <th align="left">Type</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td align="left">
            <a>
              <xsl:attribute name="href">
                #<xsl:value-of select="@ID"/>
              </xsl:attribute>
              <xsl:value-of select="Domain"/>
            </a>
          </td>
          <td align="left">
            <xsl:value-of select="Type"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="STNs">
    <div>
      <xsl:for-each select="Object">

        <xsl:if test="Type = 'eIF'">
          <xsl:for-each select="States/Object">
            <table>
              <Caption>
                <xsl:if test="position() = 1">
                  <xsl:text>if</xsl:text>
                  <xsl:text>( </xsl:text>
                  <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
                  <xsl:text> )</xsl:text>
                </xsl:if>
                <xsl:if test="(position() > 1) and (position() != last())">
                  <xsl:text>else if</xsl:text>
                  <xsl:text>( </xsl:text>
                  <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
                  <xsl:text> )</xsl:text>
                </xsl:if>
                <xsl:if test="position() = last()">
                  <xsl:text>else</xsl:text>
                </xsl:if>
              </Caption>
              <thead>
                <!--<tr>
                  <th colspan="2">
                    <xsl:if test="position() = 1">
                      <xsl:text>if</xsl:text>
                      <xsl:text>( </xsl:text>
                      <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
                      <xsl:text> )</xsl:text>
                    </xsl:if>
                    <xsl:if test="(position() > 1) and (position() != last())">
                      <xsl:text>else if</xsl:text>
                      <xsl:text>( </xsl:text>
                      <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
                      <xsl:text> )</xsl:text>
                    </xsl:if>
                    <xsl:if test="position() = last()">
                      <xsl:text>else</xsl:text>
                    </xsl:if>
                  </th>
                </tr>-->
                <tr>
                  <th align="left">Name</th>
                  <th align="left">Equation</th>
                  <!--<th align="left">Domains</th>-->
                </tr>
              </thead>
              <tbody>
                <xsl:for-each select="Equations/Object">
                  <tr>
                    <td>
                      <xsl:value-of select="Name"/>
                    </td>
                    <td>
                      <xsl:copy-of select="MathML"/>
                    </td>
                  </tr>
                </xsl:for-each>
                <!--<td>
                    <xsl:apply-templates select="STNs"/>
                  </td>-->
              </tbody>
            </table>
            <br></br>
          </xsl:for-each>

          <!--<xsl:for-each select="States/Object">
            <xsl:if test="position() = 1">
              <xsl:text>if</xsl:text>
              <xsl:text>( </xsl:text>
              <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
              <xsl:text> )</xsl:text>
            </xsl:if>
            <xsl:if test="(position() > 1) and (position() != last())">
              <xsl:text>else if</xsl:text>
              <xsl:text>( </xsl:text>
              <xsl:copy-of select="StateTransitions/Object/Condition/MathML"/>
              <xsl:text> )</xsl:text>
            </xsl:if>
            <xsl:if test="position() = last()">
              <xsl:text>else</xsl:text>
            </xsl:if>

            <xsl:apply-templates select="Equations"/>
            <xsl:apply-templates select="STNs"/>

          </xsl:for-each>-->
        </xsl:if>

        <xsl:if test="Type = 'eSTN'">
          <div>
            <xsl:for-each select="States/Object">
              <a>
                <xsl:attribute name="name">
                  <xsl:value-of select="@ID"/>
                </xsl:attribute>
                <xsl:value-of select="Name"/>
              </a>
              <br></br>
              <xsl:apply-templates select="Equations"/>
              <xsl:apply-templates select="StateTransitions"/>
              <xsl:apply-templates select="STNs"/>
            </xsl:for-each>
          </div>
        </xsl:if>

      </xsl:for-each>
    </div>
  </xsl:template>

  <xsl:template match="States">
    <div>
      <xsl:for-each select="Object">
        <a>
          <xsl:attribute name="name">
            <xsl:value-of select="@ID"/>
          </xsl:attribute>
          <xsl:value-of select="Name"/>
        </a>
        <br></br>
        <xsl:apply-templates select="Equations"/>
        <xsl:apply-templates select="StateTransitions"/>
        <xsl:apply-templates select="STNs"/>
      </xsl:for-each>
    </div>
  </xsl:template>

  <xsl:template match="StateTransitions">
    <div>
      <xsl:for-each select="Object">
        Switch to
        <a>
          <xsl:attribute name="href">
            #<xsl:value-of select="StateToRef/@ID"/>
          </xsl:attribute>
          <xsl:value-of select="Name"/>
        </a>
        if
        <xsl:apply-templates select="Condition"/>
      </xsl:for-each>
    </div>

    <!--<table border="0" cellpadding="10" cellspacing="10">
      <tr>
        <th>Name</th>
        <th>StateFrom</th>
        <th>StateTo</th>
        <th>Condition</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="StateFrom/@ID"/>
          </td>
          <td>
            <xsl:value-of select="StateTo/@ID"/>
          </td>
          <td>
            <xsl:apply-templates select="Condition"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>-->
  </xsl:template>

  <xsl:template match="Condition">
    <xsl:copy-of select="MathML"/>
  </xsl:template>

</xsl:stylesheet>
