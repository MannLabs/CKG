install.packages('BiocManager')
BiocManager::install()
BiocManager::install(c('AnnotationDbi', 'GO.db', 'preprocessCore', 'impute'))
install.packages(c('flashClust','WGCNA', 'samr'), dependencies=TRUE, repos='http://cran.rstudio.com/')
