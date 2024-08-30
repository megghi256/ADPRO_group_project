# Group_25

## Getting Started

This guide will help you set up and start using the Group_25 project. Follow the steps below to install the project and set up your environment.

## Description
The ADPRO project is a collaborative data science venture, hosted on GitLab as Group_XX, aimed at analyzing commercial flight data for sustainability studies. The team is tasked with developing a Python class adhering to PEP8 standards to handle data downloading, preprocessing, and analysis, including calculating distances using the Haversine formula and enriching the dataset with these values. An integral part of the project is the creation of a Jupyter Notebook, the showcase notebook, which presents the analysis methods and narrates the sustainability implications of commercial flights. The project emphasizes reproducibility, the importance of communication and patience in data science, and includes documentation using Sphinx, a .gitignore file, a license, and a custom README with contact information. Day two introduces advanced features such as using an LLM for aircraft and airport info and a mini-case study on decarbonization by examining the potential of replacing short-haul flights with rail services.


## Contact Information

For any inquiries or questions, please feel free to reach out to us via email:

- Margherita Thermes: 55047@novasbe.pt
- Livia Brunelli: 61715@novasbe.pt
- Luca Maniscalco: 55221@novasbe.pt


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/55047/group_25.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/55047/group_25/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)



### Setting Up the Environment

To get started, you need to set up the `adpro` environment. Ensure you have Conda installed, then run the following command in the project's root directory to create the Conda environment:

```sh
conda env create -f group25env.yml






