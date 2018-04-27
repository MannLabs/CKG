import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from plots import basicFigures
import pandas as pd
import dash_auth
import apps_config


class BasicApp:
    '''Defines what an App is in the KnowledgeViewer.
        Other Apps will inherit basic functionality from this class
        Attributes: Title, subtitle, description, logo, footer
        Functionality: setters, getters'''
    
    def __init__(self, title, subtitle, description, layout = [], logo = None, footer = None):
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.logo = logo if logo is not None else apps_config.logo
        self.footer = footer if footer is not None else apps_config.footer
        self.layout = layout
    
    #Setters
    def setTitle(self, title):
        self.title = title

    def setSubtitle(self, subtitle):
        self.subtitle = subtitle

    def setDescription(self, description):
        self.description = description

    def setLogo(self,logo):
        self.logo = logo

    def setFooter(self, footer):
        self.footer = footer

    def setLayout(self, layout):
        self.layout = layout

    def addToLayout(self, section):
        self.layout.append(section)

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

    def getHTMLDescription(self):
        return html.Div(children = self.getDescription())

    def getLogo(self):
        return self.logo

    def getFooter(self):
        return self.footer

    def getLayout(self):
        return self.layout

    #Functionality
    def addBasicLayout(self):
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
