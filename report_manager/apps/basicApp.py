from apps import apps_config as config


class BasicApp:
    '''Defines what an App is in the report_manager.
        Other Apps will inherit basic functionality from this class
        Attributes: Title, subtitle, description, logo, footer
        Functionality: setters, getters'''
    
    def __init__(self, title, subtitle, description, pageType, layout = [], logo = None, footer= None):
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.pageType = pageType
        self.logo = logo
        self.footer = footer 
        self.layout = layout
    
    #Setters
    def setTitle(self, title):
        self.title = title

    def setSubtitle(self, subtitle):
        self.subtitle = subtitle

    def setDescription(self, description):
        self.description = description

    def setPageType(self, pageType):
        self.pageType = pageType

    def setLogo(self,logo):
        self.logo = logo

    def setFooter(self, footer):
        self.footer = footer

    def setLayout(self, layout):
        self.layout = layout

    def addToLayout(self, section):
        self.layout.append(section)

    def extendLayout(self, sections):
        self.layout.extend(sections)

    #Getters
    def getTitle(self):
        return self.title

    def getHTMLTitle(self):
        return html.H1(children= self.getTitle())

    def getSubtitle(self):
        return self.subtitle

    def getHTMLSubtitle(self):
        return html.H2(children= self.getSubtitle())

    def getDescription(self):
        return self.description

    def getPageType(self):
        return self.pageType

    def getHTMLDescription(self):
        return html.Div(children = self.getDescription())

    def getFooter(self):
        return self.footer

    def getLogo(self):
        return self.logo

    def getLayout(self):
        return self.layout

    def getPageConfiguration(self):
        return config.pages[self.getPageType()]

    #Functionality
    def addBasicLayout(self):
        self.layout.append(html.Link(
            rel='stylesheet',
            href='/assests/brPBPO.css'
        ))
        if self.getTitle() is not None:
            self.layout.append(self.getHTMLTitle())
        if self.getSubtitle() is not None:
            self.layout.append(self.getHTMLSubtitle())
        if self.getDescription() is not None:
            self.layout.append(self.getHTMLDescription())
        if self.getLogo() is not None:
            self.layout.append(self.getLogo())
        if self.getFooter() is not None:
            self.layout.append(self.getFooter())

    def buildPage(self):
        self.addBasicLayout()
