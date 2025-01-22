{ pkgs }: {
  deps = [
    pkgs.python310
    (pkgs.python310.withPackages (ps: [
      ps.pip
      ps.setuptools
      ps.wheel
      ps.flask
      ps.gunicorn
    ]))
    pkgs.replitPackages.prybar-python310
    pkgs.replitPackages.stderred
  ];
  env = {
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
    STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
    PRYBAR_PYTHON_BIN = "${pkgs.replitPackages.prybar-python310}/bin/prybar-python310";
  };
}
