from apps import apps_config as config
import dash_html_components as html

class BasicApp:
    '''Defines what an App is in the report_manager.
        Other Apps will inherit basic functionality from this class
        Attributes: Title, subtitle, description, logo, footer
    '''
    
    def __init__(self, title, subtitle, description, page_type, layout = [], logo = None, footer= None):
        self._title = title
        self._subtitle = subtitle
        self._description = description
        self._page_type = page_type
        self._logo = logo
        self._footer = footer 
        self._layout = layout
    
    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title):
        self._title = title

    @property
    def subtitle(self):
        return self._subtitle

    @subtitle.setter
    def subtitle(self, subtitle):
        self._subtitle = subtitle

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, description):
        self._description = description
    
    @property
    def page_type(self):
        return self._page_type

    @page_type.setter
    def page_type(self, page_type):
        self._page_type = page_type

    @property
    def logo(self):
        return self._logo

    @logo.setter
    def logo(self, logo):
        self._logo = logo

    @property
    def footer(self):
        return self._footer

    @footer.setter
    def footer(self, footer):
        self._footer = footer

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout

    def add_to_layout(self, section):
        self.layout.append(section)

    def extend_layout(self, sections):
        self.layout.extend(sections)

    def get_HTML_title(self):
        return html.H1(children= self.title)

    def get_HTML_subtitle(self):
        return html.H2(children= self.subtitle)

    def get_HTML_description(self):
        return html.Div(children = self.description)

    #Functionality
    def add_basic_layout(self):
        self.layout.append(html.Link(
            rel='stylesheet',
            href='/assests/brPBPO.css'
        ))
        if self.title is not None:
            self.layout.append(self.get_HTML_title())
        if self.subtitle is not None:
            self.layout.append(self.get_HTML_subtitle())
        if self.description is not None:
            self.layout.append(self.get_HTML_description())
        if self.logo is not None:
            self.layout.append(self.logo)
        if self.footer is not None:
            self.layout.append(self.footer)

    def build_page(self):
        self.add_basic_layout()
