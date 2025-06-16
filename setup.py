from setuptools import setup


setup(
    name="torch-sensitivity",
    packages=[
        "tuq_main",
        "tuq_util",
        "tuq_misc",
    ],
    has_ext_modules=lambda: False,
    zip_safe=False
)
