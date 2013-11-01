#ifdef __cplusplus
extern "C" {
#endif

daeModel* CreateModel(const std::string& modelName, daeModel* parent, const std::string& description, const std::map<std::string, double>& modelOptions);

#ifdef __cplusplus
}
#endif
