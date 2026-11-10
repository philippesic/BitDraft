# 006: Comment ban and exceptions

Date: 2026-11-09. Status: foundation decision; later gates remain pending.

Full ordinary comment/docstring ban is settled. Assigned user help, regex and fixture data remain data. Markdown explanation is permitted, with fenced code checked. Only necessary shebangs, legally required license headers and genuine tool directives can receive exact path/content reasoned exceptions. No ordinary comment exception, wildcard or file-wide suppression. Unsupported syntax fails closed. Patches need complete lexical hunk context; ambiguous context fails.
