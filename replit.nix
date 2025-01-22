{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.opencv
    pkgs.pkg-config
    pkgs.libGL
    pkgs.glib
    pkgs.gtk3
    pkgs.gobject-introspection
    (pkgs.python310.withPackages (ps: [
      ps.pip
      ps.setuptools
      ps.wheel
      ps.flask
      ps.gunicorn
      ps.numpy
      ps.tensorflow
      ps.pillow
      ps.opencv4
    ]))
    pkgs.replitPackages.prybar-python310
    pkgs.replitPackages.stderred
  ];
  env = {
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
    LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
      pkgs.opencv
      pkgs.gtk3
      pkgs.glib
      pkgs.libGL
    ]}";
    STDERREDBIN = "${pkgs.replitPackages.stderred}/bin/stderred";
    PRYBAR_PYTHON_BIN = "${pkgs.replitPackages.prybar-python310}/bin/prybar-python310";
  };
}
