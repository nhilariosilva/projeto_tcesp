import setuptools

setuptools.setup(
    name = "ibgeapi",
    version = "0.0.1",
    description = "Pacote para o deflacionamento de preços pela API do IBGE",
    package_dir={'':'ibgeapi'},
    packages = setuptools.find_packages("ibgeapi")
)