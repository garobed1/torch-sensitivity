from setuptools import setup


setup(
    name="torch-sensitivity",
    packages=[
        "main_tuq",
        "util_tuq",
        "other_tuq",
    ],
    has_ext_modules=lambda: False,
    zip_safe=False
)
