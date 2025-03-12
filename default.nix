# save this as shell.nix or default.nix
{ pkgs ? import <nixpkgs> { } }:

let
  pythonPackages = pkgs.python311Packages;

  kagglehub =
    let
      pname = "kagglehub";
      version = "0.3.10";
    in
    pythonPackages.buildPythonPackage {
      inherit pname version;
      src = pkgs.fetchPypi {
        inherit pname version;
        sha256 = "7df4238eea20817bce13bfacbe79ff4c0a583a9e876bfaf16d7ad6179611fb7c";
      };
      format = "pyproject";
      nativeBuildInputs = with pythonPackages; [ hatchling ];
      propagatedBuildInputs = with pythonPackages; [
        requests
        tqdm
        packaging
        pyyaml
      ];
      doCheck = false;
    };

in
pkgs.mkShell {
  buildInputs = with pythonPackages; [
    kagglehub
  ]

  ++ (with pkgs; [
    python311
  ])

  ++ (with pythonPackages; [
    numpy
    matplotlib
    torch
    torchvision
    torchaudio
    scikit-learn
    jupyter-core
    ipykernel
    jupyterlab
    wandb
    opencv-python
    pytorch-lightning
  ]);
}
