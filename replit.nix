{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.flask
    pkgs.python39Packages.numpy
    pkgs.python39Packages.pillow
    pkgs.python39Packages.requests
  ];
  env = {
    PYTHONBIN = "${pkgs.python39}/bin/python3.9";
    LANG = "en_US.UTF-8";
    PIP_ROOT_USER_ACTION = "ignore";
  };
}
