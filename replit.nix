
{ pkgs }: {
  deps = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.flask
    pkgs.python3Packages.pillow
    pkgs.python3Packages.torch
    pkgs.python3Packages.transformers
    pkgs.python3Packages.requests
  ];
}
