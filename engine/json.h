// A small JSON reader: enough for Hugging Face config.json, model.safetensors.index.json and safetensors headers.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tuff {

struct Json {
    enum Kind { null, boolean, number, string, array, object } kind = null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool has(const std::string& k) const { return kind == object && obj.count(k); }
    const Json& operator[](const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::runtime_error("json: no key " + k);
        return it->second;
    }
    const Json& operator[](size_t i) const { return arr.at(i); }
    int i() const { return kind == boolean ? (int)b : (int)num; }
    double f() const { return num; }

    static Json parse(const std::string& s) {
        size_t p = 0;
        Json j = value(s, p);
        return j;
    }

  private:
    static void ws(const std::string& s, size_t& p) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) p++;
    }
    static std::string str_(const std::string& s, size_t& p) {
        std::string out;
        for (p++; p < s.size() && s[p] != '"'; p++) {
            if (s[p] != '\\') { out += s[p]; continue; }
            char e = s[++p];
            if (e == 'n') out += '\n';
            else if (e == 't') out += '\t';
            else if (e == 'u') { out += '?'; p += 4; }                // names here are ASCII
            else out += e;
        }
        p++;
        return out;
    }
    static Json value(const std::string& s, size_t& p) {
        ws(s, p);
        Json j;
        if (p >= s.size()) throw std::runtime_error("json: end of input");
        char ch = s[p];
        if (ch == '{') {
            j.kind = object;
            p++;
            ws(s, p);
            if (s[p] == '}') { p++; return j; }
            for (;;) {
                ws(s, p);
                std::string k = str_(s, p);
                ws(s, p);
                p++;                                                      // ':'
                j.obj[k] = value(s, p);
                ws(s, p);
                if (s[p++] == '}') return j;                              // else ','
            }
        }
        if (ch == '[') {
            j.kind = array;
            p++;
            ws(s, p);
            if (s[p] == ']') { p++; return j; }
            for (;;) {
                j.arr.push_back(value(s, p));
                ws(s, p);
                if (s[p++] == ']') return j;
            }
        }
        if (ch == '"') { j.kind = string; j.str = str_(s, p); return j; }
        if (!s.compare(p, 4, "true")) { j.kind = boolean; j.b = true; p += 4; return j; }
        if (!s.compare(p, 5, "false")) { j.kind = boolean; p += 5; return j; }
        if (!s.compare(p, 4, "null")) { p += 4; return j; }
        char* end;
        j.kind = number;
        j.num = std::strtod(s.c_str() + p, &end);
        p = end - s.c_str();
        return j;
    }
};

}  // namespace tuff
