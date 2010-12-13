<?xml version="1.0" encoding="iso-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:template match="/">
    <html>
      <body>
        <h1>
          <xsl:value-of select="Model/Name"/>
        </h1>
        <xsl:apply-templates select="Model/Domains"/>
        <xsl:apply-templates select="Model/Parameters"/>
        <xsl:apply-templates select="Model/Variables"/>
        <xsl:apply-templates select="Model/Ports"/>
        <xsl:apply-templates select="Model/Equations"/>
        <xsl:apply-templates select="Model/STNs"/>
      </body>
    </html>
  </xsl:template>

  <!--<xsl:template name="Array">
    <xsl:param name="parent"/>
    <xsl:param name="item_name"/>
    <td>
      <xsl:for-each select="$parent + '/' + $item_name">
        <xsl:value-of select="."/>
        <xsl:if test="position() != last()">
          <xsl:text>, </xsl:text>
        </xsl:if>
      </xsl:for-each>
    </td>
  </xsl:template>-->

  <xsl:template match="Domains">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Type</th>
        <th>No.Points</th>
        <th>Points</th>
        <th>LB</th>
        <th>UB</th>
        <th>DiscretizationMethod</th>
        <th>DiscretizationOrder</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="CannonicalName"/>
          </td>
          <td>
            <xsl:value-of select="Type"/>
          </td>
          <td>
            <xsl:value-of select="NumberOfPoints"/>
          </td>
            <!--<xsl:call-template name="Array">
              <xsl:with-param name="parent">
                <xsl:value-of select="."/>
              </xsl:with-param>
              <xsl:with-param name="item_name">Item</xsl:with-param>
            </xsl:call-template>-->
          <td>
            <xsl:for-each select="Points/Item">
              <xsl:value-of select="."/>
              <xsl:if test="position() != last()">
                <xsl:text>, </xsl:text>
              </xsl:if>
            </xsl:for-each>
          </td>
          <td>
            <xsl:value-of select="LeftBound"/>
          </td>
          <td>
            <xsl:value-of select="RightBound"/>
          </td>
          <td>
            <xsl:value-of select="DiscretizationMethod"/>
          </td>
          <td>
            <xsl:value-of select="DiscretizationOrder"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="Parameters">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Type</th>
        <th>Domains</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="Type"/>
          </td>
          <td>
            <xsl:for-each select="Domains/Item">
              <xsl:value-of select="."/>
              <xsl:if test="position() != last()">
                <xsl:text>, </xsl:text>
              </xsl:if>
            </xsl:for-each>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="Variables">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Type</th>
        <th>Domains</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="VariableType"/>
          </td>
          <td>
            <xsl:for-each select="Domains/Item">
              <xsl:value-of select="."/>
              <xsl:if test="position() != last()">
                <xsl:text>, </xsl:text>
              </xsl:if>
            </xsl:for-each>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="Ports">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Type</th>
        <th>Domains</th>
        <th>Parameters</th>
        <th>Variables</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="PortType"/>
          </td>
          <td>
            <xsl:apply-templates select="Domains"/>
          </td>
          <td>
            <xsl:apply-templates select="Parameters"/>
          </td>
          <td>
            <xsl:apply-templates select="Variables"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>
  
  <xsl:template match="Equations">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Domains</th>
        <th>MathML</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:apply-templates select="DistributedDomainInfos"/>
          </td>
          <td>
            <!--<xsl:apply-templates select="MathML"/>-->
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="DistributedDomainInfos">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Domain</th>
        <th>Type</th>
        <th>Points</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="Domain"/>
          </td>
          <td>
            <xsl:value-of select="Type"/>
          </td>
          <td>
            <xsl:for-each select="Points/Item">
              <xsl:value-of select="."/>
              <xsl:if test="position() != last()">
                <xsl:text>, </xsl:text>
              </xsl:if>
            </xsl:for-each>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>
  
  <xsl:template match="STNs">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Type</th>
        <th>States</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="Type"/>
          </td>
          <td>
            <xsl:apply-templates select="States"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="States">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>Equations</th>
        <th>StateTransitions</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:apply-templates select="Equations"/>
          </td>
          <td>
            <xsl:apply-templates select="StateTransitions"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>

  <xsl:template match="StateTransitions">
    <table border="1">
      <tr bgcolor="#9acd32">
        <th>Name</th>
        <th>StateTo</th>
        <th>Condition</th>
      </tr>
      <xsl:for-each select="Object">
        <tr>
          <td>
            <xsl:value-of select="Name"/>
          </td>
          <td>
            <xsl:value-of select="StateTo"/>
          </td>
          <td>
            <xsl:value-of select="Condition"/>
          </td>
        </tr>
      </xsl:for-each>
    </table>
  </xsl:template>
  
</xsl:stylesheet>