from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='DCEPredictor',
        version='1.0.2',  # 修复：verson → version
        packages=find_packages(include=['evaluater', 'trainer', 'dataset', 'utils', 'model', 
                                        'loss', 'data']),  # 强制包含你的包
        include_package_data=True,
    )
