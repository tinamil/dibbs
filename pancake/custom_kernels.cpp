
extern "C"
bool isPow2(unsigned int x)
{
  return ((x & (x - 1)) == 0);
}