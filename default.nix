# save this as shell.nix or default.nix
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    python311
  ] ++ (with python311Packages; [
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
  ]);
}
