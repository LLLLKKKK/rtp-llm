# copy from tensorflow//third_party/repo.bzl

# Copyright 2017 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining TensorFlow Bazel dependencies."""

_SINGLE_URL_WHITELIST = depset([
    "arm_compiler",
])

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _wrap_bash_cmd(ctx, cmd):
    if _is_windows(ctx):
        bazel_sh = _get_env_var(ctx, "BAZEL_SH")
        if not bazel_sh:
            fail("BAZEL_SH environment variable is not set")
        cmd = [bazel_sh, "-l", "-c", " ".join(["\"%s\"" % s for s in cmd])]
    return cmd

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Checks if we should use the system lib instead of the bundled one
def _use_system_lib(ctx, name):
    syslibenv = _get_env_var(ctx, "TF_SYSTEM_LIBS")
    if syslibenv:
        for n in syslibenv.strip().split(","):
            if n.strip() == name:
                return True
    return False

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 10)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join(cmd_and_args),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

def _repos_are_siblings():
    return Label("@foo//bar").workspace_root.startswith("../")

# Apply a patch_file to the repository root directory
# Runs 'patch -p1'
def _apply_patch(ctx, patch_file):
    # Don't check patch on Windows, because patch is only available under bash.
    if not _is_windows(ctx) and not ctx.which("patch"):
        fail("patch command is not found, please install it")
    cmd = _wrap_bash_cmd(
        ctx,
        ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)],
    )
    _execute_and_check_ret_code(ctx, cmd)

def _apply_delete(ctx, paths):
    for path in paths:
        if path.startswith("/"):
            fail("refusing to rm -rf path starting with '/': " + path)
        if ".." in path:
            fail("refusing to rm -rf path containing '..': " + path)
    cmd = _wrap_bash_cmd(ctx, ["rm", "-rf"] + [ctx.path(path) for path in paths])
    _execute_and_check_ret_code(ctx, cmd)

def _tf_http_archive(ctx):
    if ("mirror.bazel.build" not in ctx.attr.urls[0] and
        (len(ctx.attr.urls) < 2 and
         ctx.attr.name not in _SINGLE_URL_WHITELIST)):
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.bazel.build URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)
    if not use_syslib:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)

    if use_syslib and ctx.attr.system_build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.system_build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    elif ctx.attr.build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    if use_syslib:
        for internal_src, external_dest in ctx.attr.system_link_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))

tf_http_archive = repository_rule(
    implementation = _tf_http_archive,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True, allow_empty = False),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "system_build_file": attr.label(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
)

def _enable_pai(repo_ctx):
    if "TF_NEED_PAI" in repo_ctx.os.environ:
        enable_pai = repo_ctx.os.environ["TF_NEED_PAI"].strip()
        return enable_pai == "1"
    return False

def _paitf_sdk_force_rename(repo_ctx):
    if "PAITF_SDK_FORCE_RENAME" in repo_ctx.os.environ:
        paitf_sdk_force_rename = repo_ctx.os.environ["PAITF_SDK_FORCE_RENAME"].strip()
        return paitf_sdk_force_rename == "1"
    return False

def _selected_http_archive_impl(ctx):
    selected_index = 1 if (_enable_pai(ctx) or _paitf_sdk_force_rename(ctx)) else 0
    urls = [ctx.attr.urls[selected_index]]
    sha256 = ctx.attr.sha256[selected_index]
    strip_prefix = ctx.attr.strip_prefix[selected_index]

    use_syslib = _use_system_lib(ctx, ctx.attr.name)
    if not use_syslib:
        ctx.download_and_extract(
            urls,
            "",
            sha256,
            ctx.attr.type,
            strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)

    if use_syslib and ctx.attr.system_build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.system_build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    elif ctx.attr.build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    if use_syslib:
        for internal_src, external_dest in ctx.attr.system_link_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))

selected_http_archive = repository_rule(
    implementation = _selected_http_archive_impl,
    attrs = {
        "urls": attr.string_list(default = []),
        "sha256": attr.string_list(default = []),
        "strip_prefix": attr.string_list(default = []),
        "type": attr.string(),
        "delete": attr.string_list(),
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "system_build_file": attr.label(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
        "TF_NEED_PAI",
        "PAITF_SDK_FORCE_RENAME",
    ],
)

"""Downloads and creates Bazel repos for dependencies.

This is a swappable replacement for both http_archive() and
new_http_archive() that offers some additional features. It also helps
ensure best practices are followed.
"""

def _third_party_http_archive(ctx):
    if ("mirror.bazel.build" not in ctx.attr.urls[0] and
        (len(ctx.attr.urls) < 2 and
         ctx.attr.name not in _SINGLE_URL_WHITELIST)):
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.bazel.build URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)

    # Use "BUILD.bazel" to avoid conflict with third party projects that contain a
    # file or directory called "BUILD"
    buildfile_path = ctx.path("BUILD.bazel")

    if use_syslib:
        if ctx.attr.system_build_file == None:
            fail("Bazel was configured with TF_SYSTEM_LIBS to use a system " +
                 "library for %s, but no system build file for %s was configured. " +
                 "Please add a system_build_file attribute to the repository rule" +
                 "for %s." % (ctx.attr.name, ctx.attr.name, ctx.attr.name))
        ctx.symlink(Label(ctx.attr.system_build_file), buildfile_path)

    else:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)
        ctx.symlink(Label(ctx.attr.build_file), buildfile_path)

    link_dict = dict()
    if use_syslib:
        link_dict.update(ctx.attr.system_link_files)

    for internal_src, external_dest in ctx.attr.link_files.items():
        # if syslib and link exists in both, use the system one
        if external_dest not in link_dict.values():
            link_dict[internal_src] = external_dest

    for internal_src, external_dest in link_dict.items():
        ctx.symlink(Label(internal_src), ctx.path(external_dest))

# Downloads and creates Bazel repos for dependencies.
#
# This is an upgrade for tf_http_archive that works with go/tfbr-thirdparty.
#
# For link_files, specify each dict entry as:
# "//path/to/source:file": "localfile"
third_party_http_archive = repository_rule(
    implementation = _third_party_http_archive,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True, allow_empty = False),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "build_file": attr.string(mandatory = True),
        "system_build_file": attr.string(mandatory = False),
        "patch_file": attr.label(),
        "link_files": attr.string_dict(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
)
