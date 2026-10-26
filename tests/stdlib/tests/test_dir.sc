// std::path, std::dir, File::read_to_string / File::write_bytes
#include <std/test/test.h>
#include <std/path.h>
#include <std/dir.h>
#include <std/io_file.h>
#include <std/collections/string.h>
#include <std/collections/vec.h>

static void check_str(struct String s, const &char want) {
    ASSERT_STR_EQ(s.as_ptr(), want);
    s.free();
}

void t_path() {
    check_str(path_join("a", "b"), "a/b");
    check_str(path_join("a/", "b"), "a/b");
    check_str(path_join("", "b"), "b");
    check_str(path_join("a", ""), "a");
    check_str(path_join("a", "/abs"), "/abs");
    check_str(path_dirname("a/b/c"), "a/b");
    check_str(path_dirname("c"), "");
    check_str(path_dirname("/c"), "/");
    check_str(path_dirname("a/b/"), "a");
    check_str(path_basename("a/b.c"), "b.c");
    check_str(path_basename("a/b/"), "b");
    check_str(path_basename("/"), "");
    check_str(path_ext("a/b.tar.gz"), "gz");
    check_str(path_ext("b"), "");
    check_str(path_ext(".bashrc"), "");
    check_str(path_ext("a.d/b"), "");
    check_str(path_stem("a/b.tar.gz"), "b.tar");
    check_str(path_stem(".bashrc"), ".bashrc");
    check_str(path_normalize("a//b/./c/../d/"), "a/b/d");
    check_str(path_normalize("../x/../../y"), "../../y");
    check_str(path_normalize("/../a/.."), "/");
    check_str(path_normalize(""), ".");
    check_str(path_normalize("./"), ".");
    check_str(path_relative("a/b", "a/c/d"), "../c/d");
    check_str(path_relative("a/b", "a/b"), ".");
    check_str(path_relative("/x/y", "/x"), "..");
    check_str(path_relative("", "p/q"), "p/q");
    ASSERT_TRUE(path_is_absolute("/x"));
    ASSERT_FALSE(path_is_absolute("x/y"));
}

static void put(const &char path, const &char text) {
    unsigned long n = 0UL;
    unsafe { while (text[n] != '\0') { n = n + 1UL; } }
    ASSERT_TRUE(File::write_bytes(path, text, n));
}

void t_dir() {
#define ROOT "build/test_dir_tmp"
    dir_remove_all(ROOT);
    ASSERT_FALSE(path_exists(ROOT));
    ASSERT_TRUE(dir_create_all("build/test_dir_tmp/a/b/c"));
    ASSERT_TRUE(dir_create_all("build/test_dir_tmp/a/b/c"));   // already there
    ASSERT_EQ(path_kind("build/test_dir_tmp/a/b"), DIR_DIR);
    put("build/test_dir_tmp/top.txt", "top");
    put("build/test_dir_tmp/a/one.md", "# one\n");
    put("build/test_dir_tmp/a/b/c/deep.bin", "deep");
    ASSERT_TRUE(dir_create("build/test_dir_tmp/empty"));
    ASSERT_FALSE(dir_create_all("build/test_dir_tmp/top.txt/x"));  // a file is in the way

    struct FileStat st;
    ASSERT_TRUE(file_stat("build/test_dir_tmp/a/one.md", &st));
    ASSERT_EQ(st.kind, DIR_FILE);
    ASSERT_EQ(st.size, 6);
    ASSERT_TRUE(st.mtime_ns > 1600000000000000000LL);   // after 2020
    ASSERT_FALSE(file_stat("build/test_dir_tmp/missing", &st));
    ASSERT_EQ(st.kind, DIR_NONE);

    // Listing one level.
    int files = 0;
    int dirs = 0;
    ?&Dir d = Dir::open(ROOT);
    match (d) {
        case some(h): {
            struct DirEntry e = DirEntry::new();
            while (h.next(&e)) {
                if (e.kind == DIR_FILE) { files = files + 1; }
                if (e.kind == DIR_DIR) { dirs = dirs + 1; }
                ASSERT_FALSE(e.name.eq_cstr("."));
                ASSERT_FALSE(e.name.eq_cstr(".."));
            }
            e.free();
            h.close();
        }
        case null: { ASSERT_TRUE(0); }
    }
    ASSERT_EQ(files, 1);
    ASSERT_EQ(dirs, 2);
    ASSERT_TRUE(Dir::open("build/test_dir_tmp/top.txt") == null);

    // Recursive, sorted walk.
    struct Vec<struct String> all = Vec::new();
    ASSERT_TRUE(dir_walk(ROOT, &all));
    ASSERT_EQ(all.len, 3);
    ASSERT_STR_EQ(all.at(0UL).as_ptr(), "a/b/c/deep.bin");
    ASSERT_STR_EQ(all.at(1UL).as_ptr(), "a/one.md");
    ASSERT_STR_EQ(all.at(2UL).as_ptr(), "top.txt");
    all.clear();

    // Copy a tree, then read a file back whole.
    ASSERT_TRUE(dir_copy_tree("build/test_dir_tmp/a", "build/test_dir_tmp/copy/x"));
    ASSERT_TRUE(dir_walk("build/test_dir_tmp/copy", &all));
    ASSERT_EQ(all.len, 2);
    ASSERT_STR_EQ(all.at(0UL).as_ptr(), "x/b/c/deep.bin");
    all.free();
    struct String text = String::from(">");
    ASSERT_TRUE(File::read_to_string("build/test_dir_tmp/copy/x/one.md", &text));
    ASSERT_STR_EQ(text.as_ptr(), "># one\n");
    ASSERT_FALSE(File::read_to_string("build/test_dir_tmp/nope", &text));
    text.free();
    ASSERT_TRUE(file_copy("build/test_dir_tmp/top.txt", "build/test_dir_tmp/top2.txt"));
    ASSERT_TRUE(file_remove("build/test_dir_tmp/top2.txt"));
    ASSERT_FALSE(path_exists("build/test_dir_tmp/top2.txt"));
    ASSERT_TRUE(dir_remove("build/test_dir_tmp/empty"));

    ASSERT_TRUE(dir_remove_all(ROOT));
    ASSERT_FALSE(path_exists(ROOT));
    ASSERT_TRUE(dir_remove_all(ROOT));   // nothing there is fine
}

int main() {
    struct TestSuite t = TestSuite::new();
    unsafe {
        t.add("path", t_path);
        t.add("dir/file_stat/walk/copy/remove", t_dir);
    }
    t.run();
    t.print_summary();
    return t.all_passed() ? 0 : 1;
}
